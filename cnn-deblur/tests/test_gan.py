from models.deblur_wgan import DeblurWGan

deblur_gan = DeblurWGan(input_shape=(32, 32, 3))
deblur_gan.generator.summary()
deblur_gan.critic.summary()
