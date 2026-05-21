use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Tensor, backend::Backend};

pub const CRATE_NAME: &str = "sfx-models";

pub const RANGE_CHANNELS: usize = 2;
pub const RANGE_HEIGHT: usize = 64;
pub const RANGE_WIDTH: usize = 256;
pub const RGB_CHANNELS: usize = 3;
pub const RGB_HEIGHT: usize = 128;
pub const RGB_WIDTH: usize = 256;
const ENCODED_CHANNELS: usize = 64;
const RANGE_ENCODED_HEIGHT: usize = 8;
const RANGE_ENCODED_WIDTH: usize = 32;
const RANGE_ENCODED_VALUES: usize = ENCODED_CHANNELS * RANGE_ENCODED_HEIGHT * RANGE_ENCODED_WIDTH;
const RGB_ENCODED_HEIGHT: usize = 16;
const RGB_ENCODED_WIDTH: usize = 32;
const RGB_ENCODED_VALUES: usize = ENCODED_CHANNELS * RGB_ENCODED_HEIGHT * RGB_ENCODED_WIDTH;

pub fn crate_name() -> &'static str {
    CRATE_NAME
}

#[derive(Debug, Clone)]
pub struct RangeAutoencoderConfig {
    pub latent_dim: usize,
}

impl RangeAutoencoderConfig {
    pub fn new(latent_dim: usize) -> Self {
        Self { latent_dim }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> RangeAutoencoder<B> {
        RangeAutoencoder {
            enc1: downsample_conv([RANGE_CHANNELS, 16], device),
            enc2: downsample_conv([16, 32], device),
            enc3: downsample_conv([32, ENCODED_CHANNELS], device),
            to_latent: LinearConfig::new(RANGE_ENCODED_VALUES, self.latent_dim).init(device),
            from_latent: LinearConfig::new(self.latent_dim, RANGE_ENCODED_VALUES).init(device),
            dec1: upsample_conv([ENCODED_CHANNELS, 32], device),
            dec2: upsample_conv([32, 16], device),
            dec3: upsample_conv([16, RANGE_CHANNELS], device),
        }
    }
}

#[derive(Module, Debug)]
pub struct RangeAutoencoder<B: Backend> {
    enc1: Conv2d<B>,
    enc2: Conv2d<B>,
    enc3: Conv2d<B>,
    to_latent: Linear<B>,
    from_latent: Linear<B>,
    dec1: ConvTranspose2d<B>,
    dec2: ConvTranspose2d<B>,
    dec3: ConvTranspose2d<B>,
}

#[derive(Debug, Clone)]
pub struct RangeAutoencoderOutput<B: Backend> {
    pub range_hat: Tensor<B, 4>,
    pub z: Tensor<B, 2>,
}

impl<B: Backend> RangeAutoencoder<B> {
    pub fn forward(&self, range: Tensor<B, 4>) -> RangeAutoencoderOutput<B> {
        let x = relu(self.enc1.forward(range));
        let x = relu(self.enc2.forward(x));
        let x = relu(self.enc3.forward(x));

        let [batch_size, _, _, _] = x.dims();
        let x = x.reshape([batch_size, RANGE_ENCODED_VALUES]);
        let z = self.to_latent.forward(x);

        let x = relu(self.from_latent.forward(z.clone()));
        let x = x.reshape([
            batch_size,
            ENCODED_CHANNELS,
            RANGE_ENCODED_HEIGHT,
            RANGE_ENCODED_WIDTH,
        ]);
        let x = relu(self.dec1.forward(x));
        let x = relu(self.dec2.forward(x));
        let range_hat = sigmoid(self.dec3.forward(x));

        RangeAutoencoderOutput { range_hat, z }
    }
}

#[derive(Debug, Clone)]
pub struct RgbAutoencoderConfig {
    pub latent_dim: usize,
}

impl RgbAutoencoderConfig {
    pub fn new(latent_dim: usize) -> Self {
        Self { latent_dim }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> RgbAutoencoder<B> {
        RgbAutoencoder {
            enc1: downsample_conv([RGB_CHANNELS, 16], device),
            enc2: downsample_conv([16, 32], device),
            enc3: downsample_conv([32, ENCODED_CHANNELS], device),
            to_latent: LinearConfig::new(RGB_ENCODED_VALUES, self.latent_dim).init(device),
            from_latent: LinearConfig::new(self.latent_dim, RGB_ENCODED_VALUES).init(device),
            dec1: upsample_conv([ENCODED_CHANNELS, 32], device),
            dec2: upsample_conv([32, 16], device),
            dec3: upsample_conv([16, RGB_CHANNELS], device),
        }
    }
}

#[derive(Module, Debug)]
pub struct RgbAutoencoder<B: Backend> {
    enc1: Conv2d<B>,
    enc2: Conv2d<B>,
    enc3: Conv2d<B>,
    to_latent: Linear<B>,
    from_latent: Linear<B>,
    dec1: ConvTranspose2d<B>,
    dec2: ConvTranspose2d<B>,
    dec3: ConvTranspose2d<B>,
}

#[derive(Debug, Clone)]
pub struct RgbAutoencoderOutput<B: Backend> {
    pub rgb_hat: Tensor<B, 4>,
    pub z: Tensor<B, 2>,
}

impl<B: Backend> RgbAutoencoder<B> {
    pub fn forward(&self, rgb: Tensor<B, 4>) -> RgbAutoencoderOutput<B> {
        let x = relu(self.enc1.forward(rgb));
        let x = relu(self.enc2.forward(x));
        let x = relu(self.enc3.forward(x));

        let [batch_size, _, _, _] = x.dims();
        let x = x.reshape([batch_size, RGB_ENCODED_VALUES]);
        let z = self.to_latent.forward(x);

        let x = relu(self.from_latent.forward(z.clone()));
        let x = x.reshape([
            batch_size,
            ENCODED_CHANNELS,
            RGB_ENCODED_HEIGHT,
            RGB_ENCODED_WIDTH,
        ]);
        let x = relu(self.dec1.forward(x));
        let x = relu(self.dec2.forward(x));
        let rgb_hat = sigmoid(self.dec3.forward(x));

        RgbAutoencoderOutput { rgb_hat, z }
    }
}

#[derive(Debug, Clone)]
pub struct SharedLatentMultimodalAutoencoderConfig {
    pub latent_dim: usize,
    pub z_modality: usize,
}

impl SharedLatentMultimodalAutoencoderConfig {
    pub fn new(latent_dim: usize, z_modality: usize) -> Self {
        Self {
            latent_dim,
            z_modality,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SharedLatentMultimodalAutoencoder<B> {
        SharedLatentMultimodalAutoencoder {
            rgb_enc1: downsample_conv([RGB_CHANNELS, 16], device),
            rgb_enc2: downsample_conv([16, 32], device),
            rgb_enc3: downsample_conv([32, ENCODED_CHANNELS], device),
            range_enc1: downsample_conv([RANGE_CHANNELS, 16], device),
            range_enc2: downsample_conv([16, 32], device),
            range_enc3: downsample_conv([32, ENCODED_CHANNELS], device),
            rgb_to_z: LinearConfig::new(RGB_ENCODED_VALUES, self.z_modality).init(device),
            range_to_z: LinearConfig::new(RANGE_ENCODED_VALUES, self.z_modality).init(device),
            fusion_head: LinearConfig::new(self.z_modality * 2, self.latent_dim).init(device),
            shared_to_rgb: LinearConfig::new(self.latent_dim, RGB_ENCODED_VALUES).init(device),
            shared_to_range: LinearConfig::new(self.latent_dim, RANGE_ENCODED_VALUES).init(device),
            rgb_dec1: upsample_conv([ENCODED_CHANNELS, 32], device),
            rgb_dec2: upsample_conv([32, 16], device),
            rgb_dec3: upsample_conv([16, RGB_CHANNELS], device),
            range_dec1: upsample_conv([ENCODED_CHANNELS, 32], device),
            range_dec2: upsample_conv([32, 16], device),
            range_dec3: upsample_conv([16, RANGE_CHANNELS], device),
        }
    }
}

#[derive(Module, Debug)]
pub struct SharedLatentMultimodalAutoencoder<B: Backend> {
    rgb_enc1: Conv2d<B>,
    rgb_enc2: Conv2d<B>,
    rgb_enc3: Conv2d<B>,
    range_enc1: Conv2d<B>,
    range_enc2: Conv2d<B>,
    range_enc3: Conv2d<B>,
    rgb_to_z: Linear<B>,
    range_to_z: Linear<B>,
    fusion_head: Linear<B>,
    shared_to_rgb: Linear<B>,
    shared_to_range: Linear<B>,
    rgb_dec1: ConvTranspose2d<B>,
    rgb_dec2: ConvTranspose2d<B>,
    rgb_dec3: ConvTranspose2d<B>,
    range_dec1: ConvTranspose2d<B>,
    range_dec2: ConvTranspose2d<B>,
    range_dec3: ConvTranspose2d<B>,
}

#[derive(Debug, Clone)]
pub struct FusionOutput<B: Backend> {
    pub rgb_hat: Tensor<B, 4>,
    pub range_hat: Tensor<B, 4>,
    pub z_rgb: Tensor<B, 2>,
    pub z_range: Tensor<B, 2>,
    pub z_shared: Tensor<B, 2>,
}

impl<B: Backend> SharedLatentMultimodalAutoencoder<B> {
    pub fn forward(&self, rgb: Tensor<B, 4>, range: Tensor<B, 4>) -> FusionOutput<B> {
        let rgb_features = relu(self.rgb_enc1.forward(rgb));
        let rgb_features = relu(self.rgb_enc2.forward(rgb_features));
        let rgb_features = relu(self.rgb_enc3.forward(rgb_features));
        let [batch_size, _, _, _] = rgb_features.dims();
        let rgb_features = rgb_features.reshape([batch_size, RGB_ENCODED_VALUES]);
        let z_rgb = self.rgb_to_z.forward(rgb_features);

        let range_features = relu(self.range_enc1.forward(range));
        let range_features = relu(self.range_enc2.forward(range_features));
        let range_features = relu(self.range_enc3.forward(range_features));
        let range_features = range_features.reshape([batch_size, RANGE_ENCODED_VALUES]);
        let z_range = self.range_to_z.forward(range_features);

        let fused = Tensor::cat(vec![z_rgb.clone(), z_range.clone()], 1);
        let z_shared = self.fusion_head.forward(fused);

        let rgb_decoded = relu(self.shared_to_rgb.forward(z_shared.clone()));
        let rgb_decoded = rgb_decoded.reshape([
            batch_size,
            ENCODED_CHANNELS,
            RGB_ENCODED_HEIGHT,
            RGB_ENCODED_WIDTH,
        ]);
        let rgb_decoded = relu(self.rgb_dec1.forward(rgb_decoded));
        let rgb_decoded = relu(self.rgb_dec2.forward(rgb_decoded));
        let rgb_hat = sigmoid(self.rgb_dec3.forward(rgb_decoded));

        let range_decoded = relu(self.shared_to_range.forward(z_shared.clone()));
        let range_decoded = range_decoded.reshape([
            batch_size,
            ENCODED_CHANNELS,
            RANGE_ENCODED_HEIGHT,
            RANGE_ENCODED_WIDTH,
        ]);
        let range_decoded = relu(self.range_dec1.forward(range_decoded));
        let range_decoded = relu(self.range_dec2.forward(range_decoded));
        let range_hat = sigmoid(self.range_dec3.forward(range_decoded));

        FusionOutput {
            rgb_hat,
            range_hat,
            z_rgb,
            z_range,
            z_shared,
        }
    }
}

fn downsample_conv<B: Backend>(channels: [usize; 2], device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new(channels, [4, 4])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

fn upsample_conv<B: Backend>(channels: [usize; 2], device: &B::Device) -> ConvTranspose2d<B> {
    ConvTranspose2dConfig::new(channels, [4, 4])
        .with_stride([2, 2])
        .with_padding([1, 1])
        .init(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    #[test]
    fn range_autoencoder_preserves_range_shape() {
        let device = Default::default();
        let model = RangeAutoencoderConfig::new(16).init::<Flex>(&device);
        let input =
            Tensor::<Flex, 4>::zeros([2, RANGE_CHANNELS, RANGE_HEIGHT, RANGE_WIDTH], &device);

        let output = model.forward(input);

        assert_eq!(
            output.range_hat.dims(),
            [2, RANGE_CHANNELS, RANGE_HEIGHT, RANGE_WIDTH]
        );
        assert_eq!(output.z.dims(), [2, 16]);
    }

    #[test]
    fn rgb_autoencoder_preserves_rgb_shape() {
        let device = Default::default();
        let model = RgbAutoencoderConfig::new(16).init::<Flex>(&device);
        let input = Tensor::<Flex, 4>::zeros([2, RGB_CHANNELS, RGB_HEIGHT, RGB_WIDTH], &device);

        let output = model.forward(input);

        assert_eq!(
            output.rgb_hat.dims(),
            [2, RGB_CHANNELS, RGB_HEIGHT, RGB_WIDTH]
        );
        assert_eq!(output.z.dims(), [2, 16]);
    }

    #[test]
    fn shared_latent_autoencoder_preserves_rgb_and_range_shapes() {
        let device = Default::default();
        let model = SharedLatentMultimodalAutoencoderConfig::new(16, 32).init::<Flex>(&device);
        let rgb = Tensor::<Flex, 4>::zeros([2, RGB_CHANNELS, RGB_HEIGHT, RGB_WIDTH], &device);
        let range =
            Tensor::<Flex, 4>::zeros([2, RANGE_CHANNELS, RANGE_HEIGHT, RANGE_WIDTH], &device);

        let output = model.forward(rgb, range);

        assert_eq!(
            output.rgb_hat.dims(),
            [2, RGB_CHANNELS, RGB_HEIGHT, RGB_WIDTH]
        );
        assert_eq!(
            output.range_hat.dims(),
            [2, RANGE_CHANNELS, RANGE_HEIGHT, RANGE_WIDTH]
        );
        assert_eq!(output.z_rgb.dims(), [2, 32]);
        assert_eq!(output.z_range.dims(), [2, 32]);
        assert_eq!(output.z_shared.dims(), [2, 16]);
    }
}
