import paddlelite.lite as lite

opt = lite.Opt()
opt.set_model_file("inference\\mnist.pdmodel")
opt.set_param_file("inference\\mnist.pdiparams")
opt.set_optimize_out("android_model/mnist")
opt.set_valid_places("arm")
opt.set_model_type("naive_buffer")
opt.run()


