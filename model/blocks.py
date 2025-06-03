"""
Defines main blocks for ViT architecture.
"""


import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Converts HxW image into 1D embeddings.

    Splits image into patches and converts patches to 1D embeddings
    via a learnable linear layer.

    Attributes:
        patch_size: int, size of square patches to split the image into
    """


    def __init__(self,
                 patch_size: int=16,
                 in_channels: int=3,
                 embedding_dim: int=768
                 ):
        
        """Initializes patching layer.

        Intializes the convolutional and flattening operations needed to mirror
        the process of splitting an image into patches and projecting each patch
        to a 1D embedding.

        Attributes:
            p
        Args:
            patch_size: int, size of square patches to split the image into
            in_channels: int, number of color channels of input images
            embedding_dim, int, desired dimensionality of patch embeddings
        """

        super().__init__()
        self.patch_size = patch_size
        self.patchify = nn.Conv2d(in_channels=in_channels,
                                  out_channels=embedding_dim,
                                  kernel_size=patch_size,
                                  stride=patch_size)  # use convolutional layer to project patches to vectors
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3) # remove 2D structure of feature map



    def forward(self, images: torch.Tensor):
        """Executes process of converting input image to patch embeddings.

        First apply a filter over each patch of the image, then flatten 2d structure
        of feature map and re-order dimensions.

        Args:
            image: torch.Tensor, batch of input images
        Returns:
            Tensor of shape (batch_size, num_patches, embedding_dim), which is the result of
            splitting the image into patches and passing each patch through a linear layer.
        """

        image_height = images.shape[-2]
        image_width = images.shape[-1]
        assert image_height % self.patch_size == 0, "Image height must be divisible by desired patch size"
        assert image_width % self.patch_size == 0, "Image width must be divisible by desired patch size"

        # Pass patches of image through linear layer
        embeddings = self.patchify(images)

        # Flatten 2D structure of resulting feature map of convolution
        embeddings = self.flatten(embeddings)

        # Reshape resulting tensor from (batch_size, embedding_dim, num_patches) -> (batch_size, num_patches, embedding_dim)
        embeddings = embeddings.permute(0, 2, 1)

        return embeddings