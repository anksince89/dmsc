import CDMImager

dataset_name = 'kodak'
cdm_imager = CDMImager.CDMImager(dataset_name)
cdm_imager.process_images(demosaic_method='GBTF')  # You can change the method as needed

