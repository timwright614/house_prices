from house_prices_module import Model_trainer

model = Model_trainer("houses_train_raw.csv")

model.set_pipeline()

model.train_save()

print ('saved')
