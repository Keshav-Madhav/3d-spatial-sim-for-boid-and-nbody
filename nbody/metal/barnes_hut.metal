/**
 * Barnes-Hut N-Body Gravitational Force Computation - OPTIMIZED
 * ==============================================================
 * 
 * Metal compute shader optimized for Apple Silicon's Unified Memory Architecture.
 * 
 * OPTIMIZATIONS:
 * - Packed node format for better cache efficiency (48 bytes/node)
 * - Iterative traversal with fixed-size stack
 * - Vectorized SIMD operations
 * - Minimal branching for GPU efficiency
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * Packed Node structure (48 bytes, 16-byte aligned)
 * 
 * Layout matches Python numpy array packing:
 * - float4 pos_mass:   xyz = center of mass, w = total mass
 * - float4 bounds:     xyz = node center, w = half_size  
 * - int4 node_data:    x = child_base, y = body_idx, z = is_leaf, w = padding
 */
struct Node {
    float4 pos_mass;
    float4 bounds;
    int4 node_data;  // x=child_base, y=body_idx, z=is_leaf, w=unused
};

// ============================================================================
// FORCE COMPUTATION KERNEL (Barnes-Hut)
// ============================================================================

kernel void compute_forces_barnes_hut(
    device const float4* positions      [[ buffer(0) ]],
    device const Node* nodes            [[ buffer(1) ]],
    device const int* children          [[ buffer(2) ]],
    device float4* accelerations        [[ buffer(3) ]],
    device const float* params          [[ buffer(4) ]],  // [G, softening_sq, theta_sq, n_particles, n_nodes]
    uint idx                            [[ thread_position_in_grid ]]
) {
    // Load parameters
    float G = params[0];
    float softening_sq = params[1];
    float theta_sq = params[2];
    uint n_particles = uint(params[3]);
    uint n_nodes = uint(params[4]);
    
    // Bounds check
    if (idx >= n_particles) return;
    
    // Get this particle's position
    float3 pos = positions[idx].xyz;
    float3 accel = float3(0.0f);
    
    // Iterative tree traversal using stack
    int stack[64];
    int top = 0;
    stack[top++] = 0;  // Push root
    
    while (top > 0) {
        int node_idx = stack[--top];
        
        if (node_idx < 0 || node_idx >= int(n_nodes)) continue;
        
        // Load node data
        device const Node& node = nodes[node_idx];
        int is_leaf = node.node_data.z;
        int body_idx = node.node_data.y;
        
        // Skip if this is a leaf containing only this particle
        if (is_leaf && body_idx == int(idx)) continue;
        
        // Calculate vector from particle to node's center of mass
        float3 diff = node.pos_mass.xyz - pos;
        float dist_sq = dot(diff, diff) + softening_sq;
        
        // Get node size for Barnes-Hut criterion
        float node_size = node.bounds.w * 2.0f;
        float size_sq = node_size * node_size;
        
        // Barnes-Hut criterion: (s/d)² < θ²
        bool use_approximation = (is_leaf != 0) || (size_sq < theta_sq * dist_sq);
        
        if (use_approximation) {
            float mass = node.pos_mass.w;
            
            if (mass > 0.0f && dist_sq > softening_sq * 0.01f) {
                float inv_dist = rsqrt(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                float force_mag = G * mass * inv_dist3;
                accel += force_mag * diff;
            }
        } else {
            // Node too close - examine children
            int base = node.node_data.x;  // child_base
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int child = children[base + i];
                if (child >= 0 && top < 64) {
                    stack[top++] = child;
                }
            }
        }
    }
    
    accelerations[idx] = float4(accel, 0.0f);
}

// ============================================================================
// VELOCITY/POSITION UPDATE (Leapfrog Integration)
// ============================================================================

kernel void update_particles(
    device float4* positions            [[ buffer(0) ]],
    device float4* velocities           [[ buffer(1) ]],
    device const float4* accelerations  [[ buffer(2) ]],
    constant float& dt                  [[ buffer(3) ]],
    constant float& damping             [[ buffer(4) ]],
    constant uint& n_particles          [[ buffer(5) ]],
    uint idx                            [[ thread_position_in_grid ]]
) {
    if (idx >= n_particles) return;
    
    float3 pos = positions[idx].xyz;
    float3 vel = velocities[idx].xyz;
    float3 acc = accelerations[idx].xyz;
    
    // Leapfrog integration with damping
    vel = (vel + acc * dt) * damping;
    pos = pos + vel * dt;
    
    positions[idx] = float4(pos, 0.0f);
    velocities[idx] = float4(vel, 0.0f);
}

// ============================================================================
// COLOR COMPUTATION
// ============================================================================

kernel void compute_colors(
    device const float4* velocities     [[ buffer(0) ]],
    device float4* colors               [[ buffer(1) ]],
    constant float& max_speed           [[ buffer(2) ]],
    constant uint& n_particles          [[ buffer(3) ]],
    uint idx                            [[ thread_position_in_grid ]]
) {
    if (idx >= n_particles) return;
    
    float3 vel = velocities[idx].xyz;
    float speed = length(vel);
    float t = clamp(speed / max_speed, 0.0f, 1.0f);
    
    // Color gradient: bright purple-blue → blue → light blue → cyan → white → yellow → orange → red
    // Minimum brightness increased for visibility against black background
    // More color variation in slow range for better differentiation
    float3 color;
    
    if (t < 0.55f) {
        if (t < 0.15f) {
            // Bright purple-blue (0.4, 0.2, 0.8) → Blue (0.2, 0.4, 0.9)
            // Very slow bodies: bright enough to see, purple tint distinguishes them
            float s = t / 0.15f;
            color = float3(0.4f - 0.2f * s, 0.2f + 0.2f * s, 0.8f + 0.1f * s);
        } else if (t < 0.30f) {
            // Blue (0.2, 0.4, 0.9) → Light blue (0.3, 0.5, 0.95)
            float s = (t - 0.15f) / 0.15f;
            color = float3(0.2f + 0.1f * s, 0.4f + 0.1f * s, 0.9f + 0.05f * s);
        } else {
            // Light blue → Cyan → White
            float s = (t - 0.30f) / 0.25f;
            if (s < 0.6f) {
                // Light blue → Cyan
                float s2 = s / 0.6f;
                color = float3(0.3f - 0.1f * s2, 0.5f + 0.3f * s2, 0.95f + 0.05f * s2);
            } else {
                // Cyan → White
                float s2 = (s - 0.6f) / 0.4f;
                color = float3(0.2f + 0.8f * s2, 0.8f + 0.2f * s2, 1.0f);
            }
        }
    } else if (t < 0.90f) {
        // White (1.0, 1.0, 1.0) - PRIMARY RANGE
        color = float3(1.0f, 1.0f, 1.0f);
    } else if (t < 0.95f) {
        // White (1.0, 1.0, 1.0) → Yellow (1.0, 0.95, 0.0)
        float s = (t - 0.90f) / 0.05f;
        color = float3(1.0f, 1.0f - 0.05f * s, 1.0f - 1.0f * s);
    } else if (t < 0.99f) {
        // Yellow (1.0, 0.95, 0.0) → Orange (1.0, 0.5, 0.0) - RARE
        float s = (t - 0.95f) / 0.04f;
        color = float3(1.0f, 0.95f - 0.45f * s, 0.0f);
    } else {
        // Orange (1.0, 0.5, 0.0) → Red (1.0, 0.0, 0.0) - EXTREMELY RARE!
        float s = (t - 0.99f) / 0.01f;
        color = float3(1.0f, 0.5f - 0.5f * s, 0.0f);
    }
    
    colors[idx] = float4(color, 1.0f);
}

// ============================================================================
// TILED BRUTE-FORCE (for comparison/small N)
// ============================================================================

#define TILE_SIZE 256

kernel void compute_forces_tiled(
    device const float4* positions      [[ buffer(0) ]],
    device const float* masses          [[ buffer(1) ]],
    device float4* accelerations        [[ buffer(2) ]],
    device const float* params          [[ buffer(3) ]],  // [G, softening_sq, 0, n_particles, 0]
    uint idx                            [[ thread_position_in_grid ]],
    uint tid                            [[ thread_index_in_threadgroup ]],
    uint tgid                           [[ threadgroup_position_in_grid ]]
) {
    threadgroup float4 tile_pos[TILE_SIZE];
    threadgroup float tile_mass[TILE_SIZE];
    
    float G = params[0];
    float softening_sq = params[1];
    uint n = uint(params[3]);
    
    float3 pos = (idx < n) ? positions[idx].xyz : float3(0.0f);
    float3 accel = float3(0.0f);
    
    uint num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint tile = 0; tile < num_tiles; tile++) {
        uint j = tile * TILE_SIZE + tid;
        if (j < n) {
            tile_pos[tid] = positions[j];
            tile_mass[tid] = masses[j];
        } else {
            tile_pos[tid] = float4(0.0f);
            tile_mass[tid] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (idx < n) {
            for (uint k = 0; k < TILE_SIZE; k++) {
                uint j_global = tile * TILE_SIZE + k;
                if (j_global >= n || j_global == idx) continue;
                
                float3 diff = tile_pos[k].xyz - pos;
                float dist_sq = dot(diff, diff) + softening_sq;
                float inv_dist = rsqrt(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                accel += (G * tile_mass[k] * inv_dist3) * diff;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (idx < n) {
        accelerations[idx] = float4(accel, 0.0f);
    }
}
