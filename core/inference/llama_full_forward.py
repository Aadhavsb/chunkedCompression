"""
LLaMA-3 8B Full Feed-Forward Pass with Compression
Complete transformer forward pass including MLP, LayerNorm, residuals
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import time
from ..model import LLaMAModelLoader
from ..compression import LLaMACompressionProfileBuilder
from ..cache import LLaMAKVCache, StandardKVCache
from ..data import LLaMADatasetHandler

class LLaMAFullForwardInference:
    def __init__(self, model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        print(f"🚀 Initializing LLaMA-3 8B FULL Forward Pass Pipeline")
        
        # Load real LLaMA model
        self.model_loader = LLaMAModelLoader(model_path)
        from ..config import CompressionConfig
        compression_config = CompressionConfig()
        self.compression_profiles = LLaMACompressionProfileBuilder(self.model_loader, compression_config)
        self.dataset_handler = LLaMADatasetHandler(self.model_loader)
        
        # Initialize caches
        self.compressed_cache = LLaMAKVCache(enable_compression=True)
        self.standard_cache = StandardKVCache()
        
        # Model dimensions
        self.hidden_size = self.model_loader.hidden_size
        self.intermediate_size = 14336  # LLaMA-3 8B MLP dimension
        self.num_layers = self.model_loader.num_layers
        
        print(f"✅ Full forward pipeline initialized")
        print(f"   Layers: {self.num_layers}")
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   MLP intermediate: {self.intermediate_size}")

    def extract_mlp_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract MLP weights from specific layer"""
        model = self.model_loader.model
        layer = model.model.layers[layer_idx]
        
        return {
            "gate_proj": layer.mlp.gate_proj.weight,  # [intermediate_size, hidden_size]
            "up_proj": layer.mlp.up_proj.weight,      # [intermediate_size, hidden_size]
            "down_proj": layer.mlp.down_proj.weight   # [hidden_size, intermediate_size]
        }
    
    def extract_layernorm_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract layer normalization weights"""
        model = self.model_loader.model
        layer = model.model.layers[layer_idx]
        
        return {
            "input_layernorm": layer.input_layernorm.weight,      # [hidden_size]
            "post_attention_layernorm": layer.post_attention_layernorm.weight  # [hidden_size]
        }

    def mlp_forward(self, 
                   hidden_states: torch.Tensor, 
                   layer_idx: int) -> torch.Tensor:
        """
        Full MLP forward pass with SwiGLU activation
        
        Args:
            hidden_states: Input hidden states [seq_len, hidden_size]
            layer_idx: Which transformer layer
            
        Returns:
            MLP output [seq_len, hidden_size]
        """
        mlp_weights = self.extract_mlp_weights(layer_idx)
        
        # SwiGLU: gate(x) * up(x) where gate uses SiLU activation
        gate_output = F.silu(hidden_states @ mlp_weights["gate_proj"].T)  # [seq_len, intermediate_size]
        up_output = hidden_states @ mlp_weights["up_proj"].T             # [seq_len, intermediate_size]
        
        # Element-wise multiply
        intermediate = gate_output * up_output  # [seq_len, intermediate_size]
        
        # Down projection back to hidden_size
        output = intermediate @ mlp_weights["down_proj"].T  # [seq_len, hidden_size]
        
        return output

    def layer_norm(self, 
                  hidden_states: torch.Tensor, 
                  weight: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
        """RMS Layer Normalization (LLaMA uses RMSNorm)"""
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states

    def full_transformer_layer_compressed(self,
                                        hidden_states: torch.Tensor,
                                        layer_idx: int,
                                        compression_mapping: List[str]) -> torch.Tensor:
        """
        Full transformer layer with compressed attention
        
        Args:
            hidden_states: Input [seq_len, hidden_size]
            layer_idx: Layer index
            compression_mapping: Compression profile per token
            
        Returns:
            Layer output [seq_len, hidden_size]
        """
        seq_len = hidden_states.shape[0]
        layernorm_weights = self.extract_layernorm_weights(layer_idx)
        
        # 1. Input LayerNorm
        normed_input = self.layer_norm(hidden_states, layernorm_weights["input_layernorm"])
        
        # 2. Compressed Multi-Head Attention
        attention_output = torch.zeros_like(hidden_states)
        
        # Clear caches for this layer
        self.compressed_cache.clear_cache()
        
        # Compress and cache K/V for all heads
        for head_idx in range(self.model_loader.num_attention_heads):
            # Use first token as representative for caching (simplified)
            self.compress_and_cache_kv_layer(
                normed_input, compression_mapping, layer_idx, head_idx
            )
        
        # Compute attention for each position
        for pos in range(seq_len):
            query_state = normed_input[pos]  # [hidden_size]
            
            # Multi-head attention
            head_outputs = []
            for head_idx in range(self.model_loader.num_attention_heads):
                head_output = self.compressed_attention_forward_layer(
                    query_state, layer_idx, head_idx, pos
                )
                head_outputs.append(head_output)
            
            # Concatenate heads (simplified - real LLaMA does this differently)
            attention_output[pos] = torch.mean(torch.stack(head_outputs), dim=0)
        
        # 3. Residual connection
        hidden_states = hidden_states + attention_output
        
        # 4. Post-attention LayerNorm
        normed_attention = self.layer_norm(hidden_states, layernorm_weights["post_attention_layernorm"])
        
        # 5. MLP
        mlp_output = self.mlp_forward(normed_attention, layer_idx)
        
        # 6. Final residual connection
        output = hidden_states + mlp_output
        
        return output

    def full_transformer_layer_standard(self,
                                       hidden_states: torch.Tensor,
                                       layer_idx: int) -> torch.Tensor:
        """
        Full transformer layer with standard (uncompressed) attention
        """
        seq_len = hidden_states.shape[0]
        layernorm_weights = self.extract_layernorm_weights(layer_idx)
        
        # Use the real LLaMA model for standard forward pass
        model = self.model_loader.model
        layer = model.model.layers[layer_idx]
        
        with torch.no_grad():
            # This uses the actual LLaMA layer implementation
            output = layer(hidden_states.unsqueeze(0))[0]  # Add/remove batch dim
        
        return output

    def compress_and_cache_kv_layer(self,
                                   hidden_states: torch.Tensor,
                                   compression_mapping: List[str],
                                   layer_idx: int,
                                   head_idx: int):
        """Cache compressed K/V for a specific layer and head"""
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        W_K = attention_weights["W_K"]
        W_V = attention_weights["W_V"]
        
        # GQA mapping
        num_query_heads = self.model_loader.num_attention_heads
        num_kv_heads = self.model_loader.num_key_value_heads
        heads_per_kv = num_query_heads // num_kv_heads
        kv_head_idx = head_idx // heads_per_kv
        
        W_K_head = W_K[kv_head_idx * self.model_loader.head_dim:(kv_head_idx + 1) * self.model_loader.head_dim, :]
        W_V_head = W_V[kv_head_idx * self.model_loader.head_dim:(kv_head_idx + 1) * self.model_loader.head_dim, :]
        
        for token_idx, (hidden_state, profile_name) in enumerate(zip(hidden_states, compression_mapping)):
            # Project to K/V space
            key_t = W_K_head @ hidden_state
            value_t = W_V_head @ hidden_state
            
            # Compress
            compressed_key = self.compression_profiles.compress_keys(key_t, head_idx)
            compressed_value = self.compression_profiles.compress_values(value_t, profile_name, head_idx)
            
            # Store
            self.compressed_cache.store_compressed_kv(
                layer_idx=layer_idx,
                head_idx=head_idx,
                compressed_keys=compressed_key,
                compressed_values=compressed_value,
                token_idx=token_idx,
                compression_profile=profile_name
            )

    def compressed_attention_forward_layer(self,
                                         query_hidden_state: torch.Tensor,
                                         layer_idx: int,
                                         head_idx: int,
                                         position: int) -> torch.Tensor:
        """Compressed attention for specific position in layer"""
        # Project query
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        W_Q = attention_weights["W_Q"]
        W_Q_head = W_Q[head_idx * self.model_loader.head_dim:(head_idx + 1) * self.model_loader.head_dim, :]
        
        query = W_Q_head @ query_hidden_state
        
        # Get cached K/V up to current position
        cache_groups = self.compressed_cache.get_cache_groups()
        attention_output = torch.zeros(self.hidden_size, dtype=query.dtype, device=query.device)
        
        for group_layer_idx, group_head_idx, profile_name in cache_groups:
            if group_layer_idx != layer_idx or group_head_idx != head_idx:
                continue
                
            compressed_keys, compressed_values = self.compressed_cache.retrieve_compressed_kv(
                group_layer_idx, group_head_idx, profile_name
            )
            
            if compressed_keys.numel() == 0 or compressed_keys.shape[0] <= position:
                continue
            
            # Only use keys/values up to current position (causal masking)
            compressed_keys = compressed_keys[:position+1]
            compressed_values = compressed_values[:position+1]
            
            # Reconstruct and compute attention
            reconstructed_keys = self.compression_profiles.reconstruct_keys(compressed_keys, head_idx)
            scores = query @ reconstructed_keys.T
            attention_weights_tensor = F.softmax(scores, dim=0)
            context = attention_weights_tensor @ compressed_values
            
            # Decode to output space
            group_output = self.compression_profiles.decode_to_logits(context, profile_name, head_idx)
            
            # Project to hidden space (simplified)
            if group_output.shape[0] == self.model_loader.vocab_size:
                lm_head_weight = self.model_loader.get_language_model_head()
                hidden_output = group_output @ lm_head_weight / self.model_loader.vocab_size
                attention_output += hidden_output
        
        return attention_output

    def run_full_forward_benchmark(self,
                                  texts: Optional[List[str]] = None,
                                  num_layers: int = 4,
                                  max_length: int = 128) -> Dict[str, any]:
        """
        Run full forward pass benchmark through multiple transformer layers
        
        Args:
            texts: Input texts
            num_layers: Number of layers to test
            max_length: Maximum sequence length
            
        Returns:
            Comprehensive benchmark results
        """
        print(f"\n🏁 Running Full Forward Pass Benchmark")
        print(f"   Testing {num_layers} layers with max_length={max_length}")
        print(f"=" * 60)
        
        # Get real hidden states
        hidden_states_list, input_ids_list = self.dataset_handler.get_real_hidden_states_batch(
            texts, max_length
        )
        
        benchmark_results = {
            "texts_processed": len(hidden_states_list),
            "layers_tested": num_layers,
            "per_text_results": []
        }
        
        for text_idx, (hidden_states, input_ids) in enumerate(zip(hidden_states_list, input_ids_list)):
            print(f"\n--- Processing text {text_idx + 1}/{len(hidden_states_list)} ---")
            
            seq_len = hidden_states.shape[0]
            compression_mapping = self.dataset_handler.create_compression_mapping(seq_len, strategy="adaptive")
            
            # Start with input embeddings
            compressed_output = hidden_states.clone()
            standard_output = hidden_states.clone()
            
            layer_results = []
            
            # Forward through multiple layers
            for layer_idx in range(min(num_layers, self.num_layers)):
                print(f"   Layer {layer_idx + 1}/{num_layers}")
                
                # Compressed forward
                start_time = time.time()
                compressed_output = self.full_transformer_layer_compressed(
                    compressed_output, layer_idx, compression_mapping
                )
                compressed_time = time.time() - start_time
                
                # Standard forward
                start_time = time.time()
                standard_output = self.full_transformer_layer_standard(
                    standard_output, layer_idx
                )
                standard_time = time.time() - start_time
                
                # Compare outputs
                layer_mse = F.mse_loss(compressed_output, standard_output).item()
                layer_cosine = F.cosine_similarity(
                    compressed_output.flatten(), 
                    standard_output.flatten(), 
                    dim=0
                ).item()
                
                layer_results.append({
                    "layer_idx": layer_idx,
                    "output_mse": layer_mse,
                    "cosine_similarity": layer_cosine,
                    "compressed_time": compressed_time,
                    "standard_time": standard_time,
                    "speedup": standard_time / compressed_time if compressed_time > 0 else 1.0
                })
                
                print(f"     MSE: {layer_mse:.6f}, Cosine: {layer_cosine:.4f}")
                print(f"     Time: {compressed_time:.3f}s vs {standard_time:.3f}s")
            
            # Final comparison
            final_mse = F.mse_loss(compressed_output, standard_output).item()
            final_cosine = F.cosine_similarity(
                compressed_output.flatten(), 
                standard_output.flatten(), 
                dim=0
            ).item()
            
            # Memory usage
            compressed_memory = self.compressed_cache.get_memory_usage()
            standard_memory = self.standard_cache.get_memory_usage()
            
            text_results = {
                "text_index": text_idx,
                "sequence_length": seq_len,
                "final_output_mse": final_mse,
                "final_cosine_similarity": final_cosine,
                "layer_results": layer_results,
                "memory_compressed": compressed_memory["total_memory_mb"],
                "memory_standard": standard_memory["total_memory_mb"],
                "memory_savings": 1 - (compressed_memory["total_memory_mb"] / 
                                     max(standard_memory["total_memory_mb"], 1e-6))
            }
            
            benchmark_results["per_text_results"].append(text_results)
            
            print(f"   Final MSE: {final_mse:.6f}")
            print(f"   Final Cosine: {final_cosine:.4f}")
            print(f"   Memory savings: {text_results['memory_savings']:.2%}")
        
        return benchmark_results

def main():
    """Test full forward pass"""
    pipeline = LLaMAFullForwardInference()
    
    test_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Attention mechanisms enable models to focus on relevant information."
    ]
    
    results = pipeline.run_full_forward_benchmark(
        texts=test_texts,
        num_layers=2,  # Test first 2 layers
        max_length=64
    )
    
    print(f"\n🎯 FULL FORWARD PASS RESULTS")
    print(f"=" * 50)
    
    for result in results["per_text_results"]:
        print(f"Text {result['text_index'] + 1}:")
        print(f"  Final MSE: {result['final_output_mse']:.6f}")
        print(f"  Final Cosine: {result['final_cosine_similarity']:.4f}")
        print(f"  Memory savings: {result['memory_savings']:.2%}")
        
        for layer_result in result["layer_results"]:
            layer_idx = layer_result["layer_idx"]
            print(f"  Layer {layer_idx}: MSE={layer_result['output_mse']:.6f}, "
                  f"Speedup={layer_result['speedup']:.2f}x")

if __name__ == "__main__":
    main()
