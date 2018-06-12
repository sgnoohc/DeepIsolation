import keras

def simple(n_global_features): # for training with only global features
  dropout_rate = 0.0
  maxnorm = 3

  input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_1')(input_global)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_2')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_3')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_3')(deep_layer)
  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_4')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_4')(deep_layer)
  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_5')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_5')(deep_layer)
  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_6')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_6')(deep_layer)
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

  model = keras.models.Model(inputs = [input_global], outputs = [output])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
  print(model.summary())

  return model


def simple_categorized(n_kinematic_features, n_isolation_features, n_pu_features, n_ip_features):
  use_ip = False
  use_kin = True
  dropout_rate = 0.0
  maxnorm = 3

  if use_ip:
    print("using IP features")
  if use_kin: 
    print("using kinematic features")

  input_kinematic = keras.layers.Input(shape=(n_kinematic_features,), name = 'kinematic')
  input_isolation = keras.layers.Input(shape=(n_isolation_features,), name = 'isolation')
  input_pu = keras.layers.Input(shape=(n_pu_features,), name = 'pu')
  input_ip = keras.layers.Input(shape=(n_ip_features,), name = 'ip')

  deep_layer_isolation = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_1')(input_isolation)
  deep_layer_isolation = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_isolation_1')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_2')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_isolation_2')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_3')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_isolation_3')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dense(1, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_64')(deep_layer_isolation)

  deep_layer_kinematic = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_1')(input_kinematic)
  deep_layer_kinematic = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_kinematic_1')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_2')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_kinematic_2')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_3')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_kinematic_3')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dense(4, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_64')(deep_layer_kinematic)

  deep_layer_pu = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_1')(input_pu)
  deep_layer_pu = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_pu_1')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_2')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_pu_2')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_3')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_pu_3')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dense(4, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_64')(deep_layer_pu)

  deep_layer_ip = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_1')(input_ip)
  deep_layer_ip = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_ip_1')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_2')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_ip_2')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_3')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_ip_3')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dense(4, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_64')(deep_layer_ip)

  if (use_ip and use_kin):
    merged_features = keras.layers.concatenate([deep_layer_isolation, deep_layer_kinematic, deep_layer_pu, deep_layer_ip])
  elif (use_kin):
    merged_features = keras.layers.concatenate([deep_layer_isolation, deep_layer_kinematic, deep_layer_pu])
  else:
    merged_features = keras.layers.concatenate([deep_layer_isolation, deep_layer_pu])

  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_1')(merged_features)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_2')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_3')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_3')(deep_layer)

  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

  model = keras.models.Model(inputs = [input_kinematic, input_isolation, input_pu, input_ip], outputs = [output])
  optimizer = keras.optimizers.Adam()
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print(model.summary())

  return model



def categorized(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_kinematic_features, n_isolation_features, n_pu_features, n_ip_features):
  use_ip = False
  dropout_rate = 0.1
  maxnorm = 3

  input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
  input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
  input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
  input_kinematic = keras.layers.Input(shape=(n_kinematic_features,), name = 'kinematic')
  input_isolation = keras.layers.Input(shape=(n_isolation_features,), name = 'isolation')
  input_pu = keras.layers.Input(shape=(n_pu_features,), name = 'pu')
  input_ip = keras.layers.Input(shape=(n_ip_features,), name = 'ip')

  # Convolutional layers for pf cands
  dropout_rate = 0.15
  maxnorm = 3
  conv_charged_pf = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_1')(input_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_1')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(4, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_4')(conv_charged_pf)

  conv_photon_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_1')(input_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_1')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_4')(conv_photon_pf)

  conv_neutralHad_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_1')(input_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_1')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_4')(conv_neutralHad_pf)

  # LSTMs for pf cands
  batch_momentum = 0.6
  go_backwards = False

  lstm_charged_pf = keras.layers.LSTM(150, implementation = 2, name ='lstm_charged_pf_1', go_backwards = go_backwards)(conv_charged_pf)
  lstm_charged_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_charged_pf_batchnorm')(lstm_charged_pf)
  lstm_charged_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

  lstm_photon_pf = keras.layers.LSTM(100, implementation = 2, name = 'lstm_photon_pf_1', go_backwards = go_backwards)(conv_photon_pf)
  lstm_photon_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_photon_pf_batchnorm')(lstm_photon_pf)
  lstm_photon_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

  lstm_neutralHad_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_neutralHad_pf_1', go_backwards = go_backwards)(conv_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_neutralHad_pf_batchnorm')(lstm_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

  dropout_rate = 0.15
  cand_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf])
  deep_layer = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_1')(cand_features)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_2')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_3')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_3')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_4')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_4')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_5')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_5')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_6')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_6')(deep_layer)
  deep_layer = keras.layers.Dense(5, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_7')(deep_layer)

  dropout_rate = 0.1
  deep_layer_isolation = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_1')(input_isolation)
  deep_layer_isolation = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_isolation_1')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_2')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_isolation_2')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_3')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_isolation_3')(deep_layer_isolation)
  deep_layer_isolation = keras.layers.Dense(1, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_i_4')(deep_layer_isolation)

  deep_layer_kinematic = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_1')(input_kinematic)
  deep_layer_kinematic = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_kinematic_1')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_2')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_kinematic_2')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_3')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_kinematic_3')(deep_layer_kinematic)
  deep_layer_kinematic = keras.layers.Dense(1, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_k_4')(deep_layer_kinematic)

  deep_layer_pu = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_1')(input_pu)
  deep_layer_pu = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_pu_1')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_2')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_pu_2')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_3')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_pu_3')(deep_layer_pu)
  deep_layer_pu = keras.layers.Dense(1, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_pug_4')(deep_layer_pu)

  deep_layer_ip = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_1')(input_ip)
  deep_layer_ip = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_ip_1')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_2')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_ip_2')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dense(16, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_3')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_ip_3')(deep_layer_ip)
  deep_layer_ip = keras.layers.Dense(1, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_ip_4')(deep_layer_ip)

  if (use_ip):
    merged_features = keras.layers.concatenate([deep_layer, deep_layer_isolation, deep_layer_kinematic, deep_layer_pu, deep_layer_ip])
  else:
    merged_features = keras.layers.concatenate([deep_layer, deep_layer_isolation, deep_layer_kinematic, deep_layer_pu])

  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(merged_features)

  model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_kinematic, input_isolation, input_pu, input_ip], outputs = [output])
  optimizer = keras.optimizers.Adam()
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print(model.summary())

  return model

def parallel(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_global_features):
  # Inputs
  input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
  input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
  input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

  # Convolutional layers for pf cands
  dropout_rate_1 = 0.1
  maxnorm = 3
  conv_charged_pf = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_1')(input_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate_1, name = 'cpf_dropout_1')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate_1, name = 'cpf_dropout_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate_1, name = 'cpf_dropout_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(4, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_4')(conv_charged_pf)

  conv_photon_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_1')(input_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate_1, name = 'ppf_dropout_1')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate_1, name = 'ppf_dropout_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_4')(conv_photon_pf)

  conv_neutralHad_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_1')(input_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate_1, name = 'npf_dropout_1')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate_1, name = 'npf_dropout_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_4')(conv_neutralHad_pf)

  # LSTMs for pf cands
  batch_momentum = 0.6
  go_backwards = True
  print("Go backwards = " + str(go_backwards))

  lstm_charged_pf = keras.layers.LSTM(150, implementation = 2, name ='lstm_charged_pf_1', go_backwards = go_backwards)(conv_charged_pf)
  lstm_charged_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_charged_pf_batchnorm')(lstm_charged_pf)
  lstm_charged_pf = keras.layers.Dropout(dropout_rate_1, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

  lstm_photon_pf = keras.layers.LSTM(100, implementation = 2, name = 'lstm_photon_pf_1', go_backwards = go_backwards)(conv_photon_pf)
  lstm_photon_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_photon_pf_batchnorm')(lstm_photon_pf)
  lstm_photon_pf = keras.layers.Dropout(dropout_rate_1, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

  lstm_neutralHad_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_neutralHad_pf_1', go_backwards = go_backwards)(conv_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_neutralHad_pf_batchnorm')(lstm_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.Dropout(dropout_rate_1, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

  # MLP to combine LSTM outputs with global features
  dropout_rate_2 = 0.2
  cand_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf])
  deep_layer = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_1')(cand_features)
  deep_layer = keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_2')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_3')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_3')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_4')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_4')(deep_layer)
  #deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_5')(deep_layer)
  #deep_layer = keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_5')(deep_layer)
  #deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_6')(deep_layer)
  #deep_layer = keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_6')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_7')(deep_layer)

  dropout_rate_3 = 0.1
  deep_layer_global = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_g_1')(input_global)
  deep_layer_global = keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_1')(deep_layer_global)
  deep_layer_global = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_g_2')(deep_layer_global)
  deep_layer_global = keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_2')(deep_layer_global)
  deep_layer_global = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_g_3')(deep_layer_global)
  deep_layer_global = keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_3')(deep_layer_global)
  deep_layer_global = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_g_4')(deep_layer_global)
  #deep_layer_global = keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_4')(deep_layer_global)
  #deep_layer_global = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_g_5')(deep_layer_global)
  #deep_layer_global = keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_5')(deep_layer_global)
  #deep_layer_global = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_g_6')(deep_layer_global)

  dropout_rate_4 = 0.2
  merged_features = keras.layers.concatenate([deep_layer, deep_layer_global])
  deep_layer_merged = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_m_1')(merged_features)
  deep_layer_merged = keras.layers.Dropout(dropout_rate_4, name = 'mlp_dropout_m_1')(deep_layer_merged)
  deep_layer_merged = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_m_2')(deep_layer_merged)
  deep_layer_merged = keras.layers.Dropout(dropout_rate_4, name = 'mlp_dropout_m_2')(deep_layer_merged)
  deep_layer_merged = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_m_3')(deep_layer_merged)
  deep_layer_merged = keras.layers.Dropout(dropout_rate_4, name = 'mlp_dropout_m_3')(deep_layer_merged)
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer_merged)

  model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_global], outputs = [output])
  optimizer = keras.optimizers.Adam()
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print(model.summary())

  return model

def base(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_global_features):

  # Inputs
  input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
  input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
  input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

  # Convolutional layers for pf cands
  dropout_rate = 0.1
  maxnorm = 3
  conv_charged_pf = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_1')(input_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_1')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(4, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_4')(conv_charged_pf)

  conv_photon_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_1')(input_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_1')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_4')(conv_photon_pf)

  conv_neutralHad_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_1')(input_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_1')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_4')(conv_neutralHad_pf)

  # LSTMs for pf cands
  batch_momentum = 0.6
  go_backwards = False

  lstm_charged_pf = keras.layers.LSTM(150, implementation = 2, name ='lstm_charged_pf_1', go_backwards = go_backwards)(conv_charged_pf)
  lstm_charged_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_charged_pf_batchnorm')(lstm_charged_pf)
  lstm_charged_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

  lstm_photon_pf = keras.layers.LSTM(100, implementation = 2, name = 'lstm_photon_pf_1', go_backwards = go_backwards)(conv_photon_pf)
  lstm_photon_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_photon_pf_batchnorm')(lstm_photon_pf)
  lstm_photon_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

  lstm_neutralHad_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_neutralHad_pf_1', go_backwards = go_backwards)(conv_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_neutralHad_pf_batchnorm')(lstm_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

  # MLP to combine LSTM outputs with global features
  dropout_rate = 0.15
  merged_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf, input_global])
  deep_layer = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_1')(merged_features)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_2')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_3')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_3')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_4')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_4')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_5')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_5')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_6')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_6')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_7')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_7')(deep_layer)
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

  model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_global], outputs = [output])
  optimizer = keras.optimizers.Adam()
  #optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
  #optimizer = keras.optimizers.RMSprop()
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print(model.summary())

  return model

def extended_cone(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, outer_pf_timestep, n_outer_pf_features, n_global_features):
  # Inputs
  input_charged_pf = keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
  input_photon_pf = keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
  input_neutralHad_pf = keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
  input_outer_pf = keras.layers.Input(shape=(outer_pf_timestep, n_outer_pf_features), name = 'outer_pf')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'global')

  # Convolutional layers for pf cands
  dropout_rate = 0.1
  maxnorm = 3
  conv_charged_pf = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_1')(input_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_1')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_2')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Dropout(dropout_rate, name = 'cpf_dropout_3')(conv_charged_pf)
  conv_charged_pf = keras.layers.Convolution1D(4, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_charged_pf_4')(conv_charged_pf)

  conv_photon_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_1')(input_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_1')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Dropout(dropout_rate, name = 'ppf_dropout_2')(conv_photon_pf)
  conv_photon_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_photon_pf_4')(conv_photon_pf)

  conv_neutralHad_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_1')(input_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_1')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'npf_dropout_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_neutralHad_pf_4')(conv_neutralHad_pf)

  conv_outer_pf = keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_outer_pf_1')(input_outer_pf)
  conv_outer_pf = keras.layers.Dropout(dropout_rate, name = 'opf_dropout_1')(conv_outer_pf)
  conv_outer_pf = keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_outer_pf_2')(conv_outer_pf)
  conv_outer_pf = keras.layers.Dropout(dropout_rate, name = 'opf_dropout_2')(conv_outer_pf)
  conv_outer_pf = keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), activation = 'relu', name = 'conv_outer_pf_4')(conv_outer_pf)

  # LSTMs for pf cands
  batch_momentum = 0.6
  go_backwards = False

  lstm_charged_pf = keras.layers.LSTM(75, implementation = 2, name ='lstm_charged_pf_1', go_backwards = go_backwards)(conv_charged_pf)
  lstm_charged_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_charged_pf_batchnorm')(lstm_charged_pf)
  lstm_charged_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

  lstm_photon_pf = keras.layers.LSTM(50, implementation = 2, name = 'lstm_photon_pf_1', go_backwards = go_backwards)(conv_photon_pf)
  lstm_photon_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_photon_pf_batchnorm')(lstm_photon_pf)
  lstm_photon_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

  lstm_neutralHad_pf = keras.layers.LSTM(25, implementation = 2, name = 'lstm_neutralHad_pf_1', go_backwards = go_backwards)(conv_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_neutralHad_pf_batchnorm')(lstm_neutralHad_pf)
  lstm_neutralHad_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

  lstm_outer_pf = keras.layers.LSTM(75, implementation = 2, name = 'lstm_outer_pf_1', go_backwards = go_backwards)(conv_outer_pf)
  lstm_outer_pf = keras.layers.normalization.BatchNormalization(momentum = batch_momentum, name = 'lstm_outer_pf_batchnorm')(lstm_outer_pf)
  lstm_outer_pf = keras.layers.Dropout(dropout_rate, name = 'lstm_outer_pf_dropout')(lstm_outer_pf)

  # MLP to combine LSTM outputs with global features
  dropout_rate = 0.2
  merged_features = keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf, lstm_outer_pf, input_global])
  deep_layer = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_1')(merged_features)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_2')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_3')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_3')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_4')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_4')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_5')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_5')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_6')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_6')(deep_layer)
  deep_layer = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.maxnorm(maxnorm), name = 'mlp_7')(deep_layer)
  deep_layer = keras.layers.Dropout(dropout_rate, name = 'mlp_dropout_7')(deep_layer)
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer)

  model = keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_outer_pf, input_global], outputs = [output])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
  print(model.summary())

  return model

