from models.rednet import REDNet30V2

rednet = REDNet30V2((32, 32, 3))
rednet.plot_model('test.png')
