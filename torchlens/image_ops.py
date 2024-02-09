import numpy as np
import torch
import torch.nn.functional as F


def svola_convolution(image, overlap_size, psfs, psfs_grid_shape, window_type='boxcar'):
    """
        image [B, H, W, C], psfs [B, N, H, W, C] (N is the number of psfs depending on the psf grid size)
        window_type is the type of 2D separable window function used; either "boxcar" (default) or "hann"
    """
    if isinstance(overlap_size, int):
        overlap_size = (overlap_size, overlap_size)
    n_img, im_h_orig, im_w_orig, n_channels = image.shape
    n_patches, kh, kw = psfs.shape[1:4]
    im_h = im_h_orig + 2 * overlap_size[0]
    im_w = im_w_orig + 2 * overlap_size[1]
    assert kh % 2 == 1 and kw % 2 == 1
    pad_h = kh // 2
    pad_w = kw // 2
    total_pad_h = overlap_size[0] + pad_h
    total_pad_w = overlap_size[1] + pad_w

    paddings = ((0, 0), (total_pad_h, total_pad_h), (total_pad_w, total_pad_w), (0, 0))
    image = F.pad(image, paddings, mode='symmetric')

    patch_size = (im_h_orig // psfs_grid_shape[0] + overlap_size[0] * 2,
                  im_w_orig // psfs_grid_shape[1] + overlap_size[1] * 2)

    # Compute the beginning and end coordinates of all image patches
    # Padding due to overlap is considered, but not the one due to the kernel size
    # If the image shape is not a multiple of the grid shape, we stretch those coordinates
    # so that outside patches start or end at the (padded) border
    rows_0 = np.round(np.linspace(0, 1, psfs_grid_shape[0]) * (im_h - patch_size[0])).astype(int)
    cols_0 = np.round(np.linspace(0, 1, psfs_grid_shape[1]) * (im_w - patch_size[1])).astype(int)
    rows_1 = rows_0 + patch_size[0]
    cols_1 = cols_0 + patch_size[1]
    rows_0, cols_0 = np.meshgrid(rows_0, cols_0, indexing='ij')
    rows_1, cols_1 = np.meshgrid(rows_1, cols_1, indexing='ij')
    patch_corners = list(zip(rows_0.ravel(), rows_1.ravel(), cols_0.ravel(), cols_1.ravel()))

    patches = torch.stack([image[:, r0:r1 + 2 * pad_h, c0:c1 + 2 * pad_w, :] for r0, r1, c0, c1 in patch_corners], dim=0)

    # Transpose the patches from [N, B, H, W, C]
    patches = torch.transpose(patches, dim0=0, dim1=1).permute(0, 1, 3, 4, 2)  # [N, B, C, H, W]
    ph, pw = patches.shape[-2:]

    # Pad the PSFs and transpose
    psf_paddings = ((0, 0), (0, 0), (0, ph - kh), (0, pw - kw), (0, 0))
    psfs = torch.nn.functional.pad(psfs, psf_paddings, 'constant')
    psfs = psfs.permute(1, 0, 4, 2, 3)

    # Do convolution in Fourier space
    patches = patches.to(torch.complex64)
    patches = fft.fftn(patches, dim=(-3, -2))
    psfs = psfs.to(torch.complex64)
    psfs = fft.fftn(psfs, dim=(-3, -2))
    patches = patches * psfs
    patches = fft.ifftn(patches, dim=(-3, -2))
    patches = torch.abs(patches)
    patches = torch.roll(patches, shifts=[-(pad_h + 1), -(pad_w + 1)], dims=[3, 4])

    # Transpose and crop the paddings (need to reshape first)
    patches = patches.permute(0, 1, 3, 4, 2)
    patches = patches.reshape(-1, *patches.shape[-3:])
    patches = torch.nn.functional.resize_with_crop_or_pad(patches, *patch_size)
    patches = patches.reshape(n_patches, -1, *patches.shape[-3:])

    # Compute the normalized weights (contribution for each pixel of a patch to the final image)
    window_fn = {
        'boxcar': lambda x: np.ones(x.shape),
        'hann': lambda x: np.sin(np.pi * x) ** 2
    }
    row_window = window_fn[window_type](np.linspace(0, 1, patch_size[0] + 2)[1:-1])
    col_window = window_fn[window_type](np.linspace(0, 1, patch_size[1] + 2)[1:-1])
    window = row_window[:, None] * col_window[None, :]
    im_patch_weights = []
    for r0, r1, c0, c1 in patch_corners:
        im_patch_w = np.zeros((im_h, im_w, 1)).astype(np.float32)
        im_patch_w[r0:r1, c0:c1, 0] = window
        im_patch_weights.append(im_patch_w)
    normalized_weights_padded = im_patch_weights / (np.sum(np.array(im_patch_weights), axis=0))

    im_out = torch.zeros((n_img, im_h, im_w, n_channels))
    for patch, weights, (r0, r1, c0, c1) in zip(patches.unbind(), normalized_weights_padded, patch_corners):
        patch_weights = weights[r0:r1, c0:c1]
        weighted_patch = patch * torch.tensor(patch_weights)

        vertical_padding = [r0, im_h - r1]
        horizon_padding = [c0, im_w - c1]
        paddings = ((0, 0), vertical_padding, horizon_padding, (0, 0))

        padded_weighted_patch = F.pad(weighted_patch, paddings, 'constant')

        # Accumulate the results from every patch to limit memory use
        im_out = im_out + padded_weighted_patch

    im_out = im_out[:, overlap_size[0]:overlap_size[0]+im_h_orig, overlap_size[1]:overlap_size[1]+im_w_orig]
    return im_out


def repeat(x, n_repeats):
    rep = x.unsqueeze(1).repeat(1, n_repeats)
    return rep.view(-1)



import torch

def interpolate_bicubic(im, x, y, out_size):
    alpha = -0.75
    bicubic_coeffs = torch.tensor([
        [1, 0, -(alpha + 3), (alpha + 2)],
        [0, alpha, -2 * alpha, alpha],
        [0, -alpha, 2 * alpha + 3, -alpha - 2],
        [0, 0, alpha, -alpha]
    ])

    batch_size, height, width, channels = im.size()

    x = x.float()
    y = y.float()
    height_f = height.float()
    width_f = width.float()
    out_height = out_size[0]
    out_width = out_size[1]

    # Scale indices from [-1, 1] to [0, width/height - 1]
    x = torch.clamp(x, -1, 1)
    y = torch.clamp(y, -1, 1)
    x = (x + 1.0) / 2.0 * (width_f - 1.0)
    y = (y + 1.0) / 2.0 * (height_f - 1.0)

    # Do sampling
    # Integer coordinates of 4x4 neighbourhood around (x0_f, y0_f)
    x0_f = torch.floor(x)
    y0_f = torch.floor(y)
    xm1_f = x0_f - 1
    ym1_f = y0_f - 1
    xp1_f = x0_f + 1
    yp1_f = y0_f + 1
    xp2_f = x0_f + 2
    yp2_f = y0_f + 2

    # Clipped integer coordinates
    xs = [
        x0_f.long(),
        torch.max(xm1_f, torch.zeros_like(xm1_f)).long(),
        torch.min(xp1_f, width_f - 1).long(),
        torch.min(xp2_f, width_f - 1).long()
    ]
    ys = [
        y0_f.long(),
        torch.max(ym1_f, torch.zeros_like(ym1_f)).long(),
        torch.min(yp1_f, height_f - 1).long(),
        torch.min(yp2_f, height_f - 1).long()
    ]

    # Indices of neighbours for the batch
    dim2 = width
    dim1 = width * height
    base = torch.arange(batch_size) * dim1
    base = base.repeat(out_height * out_width)

    idx = []
    for i in range(4):
        idx.append([])
        for j in range(4):
            cur_idx = base + ys[i] * dim2 + xs[j]
            idx[i].append(cur_idx)

    # Use indices to lookup pixels in the flat image and restore channels dim
    im_flat = im.view(-1, channels)

    def get_weights(x, x0_f):
        tx = (x - x0_f)
        tx2 = tx * tx
        tx3 = tx2 * tx
        t = torch.stack([torch.ones_like(tx), tx, tx2, tx3])
        weights = []
        for i in range(4):
            result = torch.matmul(bicubic_coeffs[i], t)
            result = result.view(-1, 1)
            weights.append(result)
        return weights

    x_weights = get_weights(x, x0_f)
    y_weights = get_weights(y, y0_f)
    output = torch.zeros_like(im_flat)
    for i in range(4):
        x_interp = torch.zeros_like(im_flat)
        for j in range(4):
            # To calculate interpolated values first, interpolate in x dim 4 times for y=[0, -1, 1, 2]
            x_interp = x_interp + x_weights[j] * im_flat[idx[i][j]]
        # Finally, interpolate in y dim using interpolations in x dim
        output = output + y_weights[i] * x_interp

    output = output.view(batch_size, out_height, out_width, channels)
    return output

