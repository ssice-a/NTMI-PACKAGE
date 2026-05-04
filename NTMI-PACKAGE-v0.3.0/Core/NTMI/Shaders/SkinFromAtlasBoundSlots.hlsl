// Skin mesh buffers directly from the collector-built global T0 atlas.
//
// t64 = global T0 atlas, 3 uint4 rows per global bone
// t65 = local palette, palette[localBone] = globalBone
// t66 = packed blend buffer exposed as R32_UINT, 2 uints per vertex
// t67 = source frame buffer, 2 snorm float4 rows per vertex
// t68 = source position buffer, 3 floats per vertex
// u6  = skinned frame/normal UAV, 2 snorm float4 rows per vertex
// u7  = skinned position UAV, 3 floats per vertex

StructuredBuffer<uint4> PoseGlobalT0 : register(t64);
Buffer<uint> LocalPalette : register(t65);
Buffer<uint> BlendTyped : register(t66);
Buffer<float4> SourceFrame : register(t67);
Buffer<float> SourcePosition : register(t68);

RWBuffer<float4> ScratchFrameUAV : register(u6);
RWBuffer<float> ScratchPositionUAV : register(u7);

uint local_vertex_count()
{
    uint position_float_count = 0u;
    SourcePosition.GetDimensions(position_float_count);
    return position_float_count / 3u;
}

uint local_bone_count()
{
    uint count = 0u;
    LocalPalette.GetDimensions(count);
    return count;
}

uint global_bone_count()
{
    uint rows = 0u;
    uint stride = 0u;
    PoseGlobalT0.GetDimensions(rows, stride);
    return rows / 3u;
}

bool load_transform(uint local_bone, uint local_count, uint global_count,
        out float4 row0, out float4 row1, out float4 row2)
{
    row0 = 0.0;
    row1 = 0.0;
    row2 = 0.0;

    if (local_bone >= local_count || global_count == 0u)
    {
        return false;
    }

    uint global_bone = min(LocalPalette[local_bone], global_count - 1u);
    uint row_base = global_bone * 3u;
    row0 = asfloat(PoseGlobalT0[row_base + 0u]);
    row1 = asfloat(PoseGlobalT0[row_base + 1u]);
    row2 = asfloat(PoseGlobalT0[row_base + 2u]);
    return true;
}

float3 transform_point(float4 row0, float4 row1, float4 row2, float3 position)
{
    return float3(
        dot(row0.xyz, position) + row0.w,
        dot(row1.xyz, position) + row1.w,
        dot(row2.xyz, position) + row2.w
    );
}

float3 transform_vector(float4 row0, float4 row1, float4 row2, float3 vector_value)
{
    return float3(
        dot(row0.xyz, vector_value),
        dot(row1.xyz, vector_value),
        dot(row2.xyz, vector_value)
    );
}

float3 normalize_or(float3 value, float3 fallback)
{
    float length_sq = dot(value, value);
    if (length_sq <= 1.0e-12)
    {
        return fallback;
    }
    return value * rsqrt(length_sq);
}

void accumulate_influence(uint local_bone, float weight, uint local_count, uint global_count,
        float3 source_position, float3 source_frame0, float3 source_frame1,
        inout float3 skinned_position, inout float3 skinned_frame0, inout float3 skinned_frame1)
{
    if (weight <= 0.0)
    {
        return;
    }

    float4 row0, row1, row2;
    if (!load_transform(local_bone, local_count, global_count, row0, row1, row2))
    {
        return;
    }

    skinned_position += transform_point(row0, row1, row2, source_position) * weight;
    skinned_frame0 += transform_vector(row0, row1, row2, source_frame0) * weight;
    skinned_frame1 += transform_vector(row0, row1, row2, source_frame1) * weight;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint vertex_id = tid.x;
    uint vertex_count = local_vertex_count();
    if (vertex_id >= vertex_count)
    {
        return;
    }

    uint local_count = local_bone_count();
    uint global_count = global_bone_count();

    uint blend_base = vertex_id * 2u;
    uint packed_indices = BlendTyped[blend_base + 0u];
    uint packed_weights = BlendTyped[blend_base + 1u];

    uint4 indices = uint4(
        packed_indices & 0xffu,
        (packed_indices >> 8u) & 0xffu,
        (packed_indices >> 16u) & 0xffu,
        (packed_indices >> 24u) & 0xffu
    );
    float4 weights = float4(
        (float)(packed_weights & 0xffu),
        (float)((packed_weights >> 8u) & 0xffu),
        (float)((packed_weights >> 16u) & 0xffu),
        (float)((packed_weights >> 24u) & 0xffu)
    ) * (1.0 / 255.0);

    float weight_sum = weights.x + weights.y + weights.z + weights.w;

    uint position_base = vertex_id * 3u;
    float3 source_position = float3(
        SourcePosition[position_base + 0u],
        SourcePosition[position_base + 1u],
        SourcePosition[position_base + 2u]
    );

    uint frame_base = vertex_id * 2u;
    float4 source_frame0 = SourceFrame[frame_base + 0u];
    float4 source_frame1 = SourceFrame[frame_base + 1u];

    float3 skinned_position = source_position;
    float3 skinned_frame0 = source_frame0.xyz;
    float3 skinned_frame1 = source_frame1.xyz;

    if (weight_sum > 1.0e-6)
    {
        weights /= weight_sum;

        skinned_position = 0.0;
        skinned_frame0 = 0.0;
        skinned_frame1 = 0.0;

        accumulate_influence(indices.x, weights.x, local_count, global_count,
                source_position, source_frame0.xyz, source_frame1.xyz,
                skinned_position, skinned_frame0, skinned_frame1);
        accumulate_influence(indices.y, weights.y, local_count, global_count,
                source_position, source_frame0.xyz, source_frame1.xyz,
                skinned_position, skinned_frame0, skinned_frame1);
        accumulate_influence(indices.z, weights.z, local_count, global_count,
                source_position, source_frame0.xyz, source_frame1.xyz,
                skinned_position, skinned_frame0, skinned_frame1);
        accumulate_influence(indices.w, weights.w, local_count, global_count,
                source_position, source_frame0.xyz, source_frame1.xyz,
                skinned_position, skinned_frame0, skinned_frame1);

        skinned_frame0 = normalize_or(skinned_frame0, source_frame0.xyz);
        skinned_frame1 = normalize_or(skinned_frame1, source_frame1.xyz);
    }

    ScratchPositionUAV[position_base + 0u] = skinned_position.x;
    ScratchPositionUAV[position_base + 1u] = skinned_position.y;
    ScratchPositionUAV[position_base + 2u] = skinned_position.z;

    ScratchFrameUAV[frame_base + 0u] = float4(skinned_frame0, source_frame0.w);
    ScratchFrameUAV[frame_base + 1u] = float4(skinned_frame1, source_frame1.w);
}
