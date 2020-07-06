import os
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch.nn.functional as F
import torch.optim as optim

def download_pretrained_model(file_id, path, unzip=True, overwrite=False):
  """ Downloads pretrained model to the given path.

      file_id -- file id stored in google drive
      path -- path that the file is to be stored
      unzip -- unzip the file 
      overwrite -- overwrite if the file exists
      
      example file_id: 
      https://drive.google.com/open?id=1-7jiTlEvt1_G2dOK4JZiYmE4uJhkD7l8
      or just 1-7jiTlEvt1_G2dOK4JZiYmE4uJhkD7l8
  """
  
  if "drive.google.com" in file_id:
    idx = file_id.find("id=")
    file_id = file_id[idx+3:]
    print("file_id fetched from url: ", file_id)

  # check whether file exists
  if os.path.exists(path) and not overwrite:
    print("Model is already downloaded. path: ", path)
    return

  folder = os.path.dirname(path)
  if not os.path.exists(folder):
    os.makedirs(folder)
  gdd.download_file_from_google_drive(file_id=file_id, dest_path= path, 
                                      unzip=unzip, overwrite=overwrite)
                                      
                                      
def initialize_weights(m, config):    
  """ Initializes the weights with given constraints.
      m -- pytorch entity to initialize
      config -- configs used in initialization 
  """
  if hasattr(m, 'weight') and m.weight is not None:
    # initialize weights with normal distribution
    torch.nn.init.normal_(m.weight.data, 
                        config["weight_initialization_mean"], 
                        config["weight_initialization_var"])
  if hasattr(m, 'bias') and  m.bias is not None:
    # initialize bias with a constant value
    torch.nn.init.constant_(m.bias.data, 
                            config["bias_initialization"])
                            
def draw_scatter_plot(G, config, synthetic_data, device, ax, plot_together=False):
  """ Generates samples from the generator and draws a scatter plot containing 
      both real and fake samples.
  """
  batch_size_for_gen = config["num_gens"] * config["gen_batch_size"]
  latent = torch.randn(batch_size_for_gen, config["z_length"], device=device)
  points = G(latent)

  points = points.detach().to("cpu").numpy()

  ax.scatter(synthetic_data[:, 0], synthetic_data[:, 1], label="real data", c="red")
  
  if plot_together:
    ax.scatter(points[:, 0], points[:, 1], label="generated data", c="blue")
  else:
    for i in range( config["num_gens"]):
      gen_data = points[i*config["gen_batch_size"]: (i+1)*config["gen_batch_size"], :]
      ax.scatter(gen_data[:, 0], gen_data[:, 1], label="gen {}".format(i))


  ax.legend(loc="upper right", markerscale=2)
  
def symmetric_KL_distance(x1, x2):
  if x1.shape != x2.shape:
    raise Exception("x1 {} and x2 {} dimensions must be equal!".format(P.shape, Q.shape))

  x1 = torch.Tensor(x1)
  x2 = torch.Tensor(x2)
  out = F.kl_div(x1, x2) + F.kl_div(x2, x1)
  return float(out) * 0.5
  

def save_model(path, G, D, opt1, opt2, epoch, batch_offset=0, 
               config=None, verbose=False, kl_distances=[]):
  """ Saves model to the given path as checkpoint. 
  """

  if verbose:
    print("Saving model path: ", path)
  
  # check folder exists
  folder = os.path.dirname(path)
  if not os.path.exists(folder):
    os.makedirs(folder)

  # saves pytorch checkpoint    
  torch.save({"epoch": epoch,
              "G": G.state_dict(),
              "D": D.state_dict(),
              "opt1": opt1.state_dict(),
              "opt2": opt2.state_dict(),
              "config": config,
              "batch_offset": batch_offset,
              "kl_distances": kl_distances
  }, path)

def find_last_model_path(config):
  """ Util function to continue training from last model.
  """
  model_name_format = "{datatype}_numgen{num_gens:02d}_beta{rbeta:02d}"
  model_name = model_name_format.format(**config, 
                                        rbeta=int(config["regularization_beta"]*100))
  
  print("Checking model_name: ", model_name, " searching in ", config["model_save_dir"])
  files = []
  for (dirpath, dirnames, filenames) in os.walk(config["model_save_dir"]):
    files += [os.path.join(dirpath, file) for file in filenames if file.startswith(model_name)]
  files.sort()
  if files:
    print("Latest model path is ", files[-1])
  else:
    print("No model found")
  
  return None if not files else files[-1]                                       
