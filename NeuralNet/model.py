import keras

def base(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_global_features):

  # Inputs
  input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
  input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
  input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

  # Convolutional layers for pf cands
  dropout_rate = 0.1
  conv_charged_pf = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_1')(input_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_1')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(4, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_charged_pf_4')(conv_charged_pf)

  conv_photon_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_photon_pf_1')(input_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_1')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_photon_pf_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_photon_pf_4')(conv_photon_pf)

  conv_neutralHad_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_neutralHad_pf_1')(input_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_1')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_neutralHad_pf_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_neutralHad_pf_4')(conv_neutralHad_pf)

  # LSTMs for pf cands
  batch_momentum = 0.6

  lstm_charged_pf = keras.layers.LSTM(150, implementation = 2, name ='lstm_charged_pf_1')(conv_charged_pf)
  lstm_charged_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_charged_pf_batchnorm')(lstm_charged_pf)
  lstm_charged_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

  lstm_photon_pf = keras.layers.LSTM(100, implementation = 2, name = 'lstm_photon_pf_1')(conv_photon_pf)
  lstm_photon_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_photon_pf_batchnorm')(lstm_photon_pf)
  lstm_photon_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

  lstm_neutralHad_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_neutralHad_pf_1')(conv_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_neutralHad_pf_batchnorm')(lstm_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

  # MLP to combine LSTM outputs with global features
  dropout_rate = 0.15
  merged_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf, input_global])
  deep_layer = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_1')(merged_features)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_2')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_3')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_3')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_4')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_4')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_5')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_5')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_6')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_6')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'mlp_7')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_7')(deep_layer)
  output = keras.layers.Dense(1, activation = 'tanh', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

  model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_global], outputs = [output])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

  return model 
