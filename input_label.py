img = cv.imread("images/seagull_database_vis002_small.png")


windowsize_r = size_block(res)[0]
windowsize_c = size_block(res)[1]

original_height = image.shape[0]
original_width = image.shape[1]

groundtruth = np.zeros((original_width/windowsize_c*original_height/windowsize_r,windowsize_r,windowsize_c))




