from ..models.ms_deblur_wgan import MSDeblurWGAN

deblur_gan = MSDeblurWGAN(input_shape=(32, 32, 3))
deblur_gan.generator.summary()
deblur_gan.critic.summary()
