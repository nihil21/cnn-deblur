from models.deblur_gan import DeblurGan

deblur_gan = DeblurGan(input_shape=(32, 32, 3))
deblur_gan.generator.summary()
deblur_gan.critic.summary()
