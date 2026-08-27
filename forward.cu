/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h" // contains the precalculated orthonormalized SH coefficients
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// this code is the same logic as utils/sh_utils.py and gaussian_renderer/__init__.py
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022) https://github.com/sjtuzq/point-radiance/blob/main/modules/sh.py
	glm::vec3 pos = means[idx]; // get 3D gaussian mean by indexing the point
	glm::vec3 dir = pos - campos; // campos is camera center: dir = center of 3D Gaussian - center of cam-center  {vec c = vec a - vec b}

	dir = dir / glm::length(dir); // normalize viewing direction vector

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs; // pointer to the SH coefficients for this specific Gaussian
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0) // deg 1 : xyz components for sH
	{
		float x = dir.x; // separate the normalized viewing direction components x
		float y = dir.y; // separate the normalized viewing direction components y
		float z = dir.z; // separate the normalized viewing direction components z
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]; //constant * view_dir * sh_learned_coeff

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	// From sh_utils.py
	// we multiply the learnt coefficient sh with base 0 constant & add 0.5
	// dc sh coefficients are initialized from data so they are not 0
	// however, sH coefficients are normalized so a resultant effect from sH if 0, lands on gray (by convention)
	result += 0.5f; // here results is sh * C0

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass
	/
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// Cov2D = J * W * Cov3D * transpose (W) * transpose(J)
	float3 t = transformPoint4x3(mean, viewmatrix); // take the xyz location of 3D gaussians and transform to camera view (3D space)

	const float limx = 1.3f * tan_fovx; // tan_fovx = X/Z
	const float limy = 1.3f * tan_fovy; // tan_fovx = X/Z; gaussians near camera plane 
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z; // just measuring t.x
	t.y = min(limy, max(-limy, tytz)) * t.z; // just measuring t.y
	/*
		J = [[f_x/z   0    -f_x * x /z**2]
             [  0   f_y/z   -f_y * y /z**2]
			 [  0   0   0] ]
	
	*/
	glm::mat3 J = glm::mat3( 
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0); // 3x3 glm matrix for jacobian : a transformation from cameras 3Dspace to ray space {projective mapping}
	
	// viewmatrix is viewing transfomration:from world sys to camera's own 3D system, viewmatrix is 4X4
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]); // column-major [V00, V10, V20, V01, V11, V21, V02, V12, V22] (arranged like an array)

	glm::mat3 T = W * J; // T = Jacobian * viewing transformation (glm is right multiplications)

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]); // 3D covariance matrix

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) }; // 2D covar is 2x2 MAT, and also +ve-semi-definite so entry [1][0] ignored
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D) // Cov3D = R*S*transpose(S)*transpose(R)
{
	// Create scaling matrix (3x3 triangular matrix); mod: scale_modifier = kept as 1.0 during training
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation; already normalized
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x; // r is the scalar part; x,y,z are complex components
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion, same formula as build_rotation function in utils/general_utils
	// This formula https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation (right handed rule)
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R; // in paper its R*S, in glm matrix multiplication is Right->left, so it is (RS)
	// Paper Formula: R*S*transpose(S)*transpose(R) = (RS)*Transpose(RS)
	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M; // its means M*transpose(M) where (RS)*tranpose(RS) ; this mult op guarantees symmetry & PSD


	// since cov3D is perfectly symmetric, the upper triangular values
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization. // this is the processing from 3D to image pixel colors
template<int C>
__global__ void preprocessCUDA(int P, int D, int M, // P: num gaussians, D: sH deg, M: total number of SH coefficients per gaussian (e.g. 16 for deg 3)
	const float* orig_points, // original 3D points (mean3D) [N,3]
	const glm::vec3* scales, // scale vectors [N,3]
	const float scale_modifier, // value is 1.0
	const glm::vec4* rotations, // quaternion vectors [N,(r,x,y,z)]
	const float* opacities, // array of opacities [N]
	const float* shs, // array of sHs coeff
	bool* clamped,
	const float* cov3D_precomp, // precomputed Cov3D (if exists) of each 3D gaussian
	const float* colors_precomp, // precomputed colors (if exists) of each 3D gaussian
	const float* viewmatrix, // 4X4 3dworld to 3dcamera viewmatrix [R|t] stored as an array 
	const float* projmatrix, // 4X4 (world-to-viewspace) matrix
	const glm::vec3* cam_pos, // xyz cam center
	const int W, int H, // Img width and height in pixels
	const float tan_fovx, float tan_fovy, // tan of field of views
	const float focal_x, float focal_y, // focal lengths of image
	int* radii, // radii of 2D gaussians
	float2* points_xy_image, // center of 2D gaussians
	float* depths,
	float* cov3Ds, // Gaussians cov3D matrices
	float* rgb, // Each gaussian's rgb output buffer
	float4* conic_opacity, // the inverse symmtric cov2D matrix (xx,xy,yy) along with the opacity 
	const dim3 grid,
	uint32_t* tiles_touched, // for each indexed gaussian get the tileIDs it overlaps
	bool prefiltered, // check if gaussian is already filtered for frustum-culling
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank(); // thread indexing
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0; // set the radii of the 3D gaussian at idx to 0
	tiles_touched[idx] = 0; // a counter to check the number of tiles it touches; precursor for duplication in rasterizer_impl.cu

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) // dont process gaussians outside view-frustum
		return;

	// Transform point by projecting to 2D screen space
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] }; // 3D mean of gaussians; array is flattened so idx*3
	float4 p_hom = transformPoint4x4(p_orig, projmatrix); // transform the 3d mean from world to 2D viewspace homogenous coordinates
	float p_w = 1.0f / (p_hom.w + 0.0000001f); // convert to  (NDC coords) [-1,1] and avoid 0-division error
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr) // if Cov3D provided
	{
		cov3D = cov3D_precomp + idx * 6; // 6 because of 6 values (00,01,02,11,12,22) for semi-positive definite matrix
	}
	else // if Cov3D not provided, as standard case
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6); // compute the matrix cov3d
		cov3D = cov3Ds + idx * 6; // cov3D is a contiguous array for all cov3Ds of gaussians, each indexed by idx
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	/*
	If a Gaussian is smaller than a pixel, it causes aliasing
	cov2d = (sigma_xx, sigma_xy, sigma_yy) == (cov.x, cov.y, cov.z)
	enlarge it by adding a variance of 0.3 to the diagonal
	*/
	constexpr float h_var = 0.3f; // heuristic variance adder;  adding some positive value to the diagonal means adding λI
	// make it invertible; positive definite 
	const float det_cov = cov.x * cov.z - cov.y * cov.y; // cov2D determinant
	cov.x += h_var; // add the variance to sigma_x
	cov.z += h_var; // add the variance to sigma_y
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y; // update the determinant
	float h_convolution_scaling = 1.0f;

	if(antialiasing) // h_convolution_scaling is opacity scaling factor, reduces opacity since enlarging gaussians area should distribute opacity more
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm) 
	const float det = det_cov_plus_h_cov;
	// https://github.com/kwea123/gaussian_splatting_notes 
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det; // do the inversion to get impact of each gaussian on a pixel
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv }; // inverse of 2d covariance matrix
	// https://medium.com/data-science/a-python-engineers-introduction-to-3d-gaussian-splatting-part-2-7e45b270c1df
	// Compute extent in screen space (by finding eigenvalues)
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z); // mid is the 0.5*trace of the 2x2 matrix
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); // eigenvalue1 radii of the projected ellipsis along major/minor axis
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det)); // eigenvalue2 radii of the projected ellipsis along major/minor axis
	// 3 times sqrt of largest eignvalue represents 3 standard deviation = covers 99.7% of projected 2D gaussian distribution
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); // compute the extent in screenspace
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) }; // convert NDC -> pixel space for gaussians
	uint2 rect_min, rect_max; // initialize the rectangle
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) // getRect calculates the bounding box of tiles that the radius (splat) covers
		return; // Cull if it doesn't intersect any tiles ( outside the frustum)

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	} // compute per channel using per-channel resultant SH coefficient effect 

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;  // The depth in camera space (used as the sorting key later)
	radii[idx] = my_radius; // The radius is in pixel
	points_xy_image[idx] = point_image; //center of the Gaussian in 2D pixel coordinate
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling }; // opacity scaled for anti-aliasing with h_convolution_scaling, which is otherwise 1


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);// save number of tiles this gaussian overlaps
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS> // each thread treat 1 tile
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list, // sorted Gaussian IDs
	int W, int H,
	const float2* __restrict__ points_xy_image, // screen-space centers (x,y) of all Gaussians
	const float* __restrict__ features, // RGB colors
	const float4* __restrict__ conic_opacity, // inverse 2D covariance and opacity
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color, // Background color
	float* __restrict__ out_color, // final rendered RGB image
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // number of tiles that cover the width of the image
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };// top-left pixel coordinate (x,y) of the current tile
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) }; // bottom-right pixel coordinate (x,y) of the current tile
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list. 
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x; // total # of Gaussians overlapping this tile

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f; // transmittance starts at 1
	uint32_t contributor = 0;
	uint32_t last_contributor = 0; 
	float C[CHANNELS] = { 0 }; // accumulated RGB color for this thread's pixel

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress]; // collect id of gaussian
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; // collect its pixel location and conic opacity
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch 
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// Read the center (xy) and compute the 2D offset (d) from the Gaussian center to pixel.
			float2 xy = collected_xy[j]; //  xy: the 2d coord of the Gaussian center
			float2 d = { xy.x - pixf.x, xy.y - pixf.y }; // pixf: the 2d coord of the current pixel; d is the distance at pixel level
			float4 con_o = collected_conic_opacity[j]; // // con_o: inv cov2d (x,y,z), opacity (w)
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity (contribution from opacity)
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) // if contribution < 1/255 skip that gaussian contribution
				continue;
			
			// alpha-composition: front to back
			// T_new = T_old * (1 - alpha)
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) // a check to stop alpha blending if saturation is reached
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// save the final transmittance and the index of the last contributing Gaussian
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		
		// Blend the accumulated color w. bg color using the transmittance
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
