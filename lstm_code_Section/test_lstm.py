from subprocess import call
call(["python","\lstm_code_Section\saliency_lstm_training_test.py",
       "--data", "features",
        '--lr', '0.01',
        '--epochs', '100',

        '--valgrid', 'grids/grid1616_no_norm/val_grid.txt',
        '--testgrid', 'grids/grid1616_no_norm/test_grid.txt',
        '--traingrid', 'grids/grid1616_no_norm/train_grid.txt', 
        '--gazemaps', 'raw_data/test/gazemap_images', 
        '--yolo5bb', 'runs/detect/exp2/labels', 
        '--visualizations', 'grid1616', 
        '--threshhold', '0.5', 

        '--best', 'grid1616_lstm_model_best_anuj.pth.tar' ])