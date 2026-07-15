from ....pytorch_abc import PyTorchAbstractClass 
import importlib
import os
import torch
import torchaudio
from pyannote.audio import Pipeline

class PyTorch_Pyannote_Diarization(PyTorchAbstractClass):
  def __init__(self, config=None):
    # 1. Initialize the parent class to set up devices and config
    super().__init__(config)
    
    # 2. Authenticate with Hugging Face (required for gated pyannote models)
    try:
        self.huggingface_authenticate()
    except Exception as e:
        print(f"Warning: Hugging Face authentication failed: {e}. "
              "Ensure HUGGINGFACE_TOKEN is set in your environment.")

    # Get the token to pass directly to the loader as well
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or self.config.get('huggingface_token')

    # 3. Load the Hugging Face model
    self.model = self.load_hf_model(
        model_class=Pipeline, 
        model_name_or_path="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    if self.model is None:
        raise RuntimeError(
            "Failed to load PyAnnote Pipeline. Verify your HUGGINGFACE_TOKEN "
            "is valid and that you have accepted the user conditions for "
            "'pyannote/speaker-diarization-3.1' on Hugging Face."
        )

    # 4. Handle PyAnnote's native device mapping 
    device = torch.device(self._device)
    try:
        self.model.to(device)
    except Exception as e:
        print(f"Warning: Could not natively move PyAnnote pipeline to {device}: {e}")

    # 5. Prevent the framework runner from trying to run standard PyTorch .to() dispatch
    self._is_dispatched = True

  def eval(self):
    # Overridden as a no-op because pyannote.audio.Pipeline is not an nn.Module 
    # and does not implement or require .eval()
    pass

  def to(self, device, multi_gpu=False):
    # Overridden to prevent standard PyTorch model shifting, 
    # as pyannote handles device routing internally.
    self.device = device
    self._is_dispatched = True
    return self

  def preprocess(self, input_audios):
    # input_audios is a list of file paths
    print("Preprocessing audio files...")
    preprocessed_inputs = []
    
    # Note: fixed 'input_data' to 'input_audios' reference here
    for file_path in input_audios:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample to 16,000 Hz if necessary (PyAnnote models expect 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Convert multi-channel (stereo) audio to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        preprocessed_inputs.append({
            "waveform": waveform, 
            "sample_rate": 16000
        })
            
    return preprocessed_inputs

  def predict(self, model_input): 
      print("Running diarisation model predictions...")
      predictions = []
      
      # Switch model to evaluation mode (using our safe overridden method)
      self.eval()
      
      with torch.no_grad():
          for audio_payload in model_input:
              # Run the pipeline inference
              annotation = self.model(audio_payload)
              predictions.append(annotation)
              
      return predictions

  def postprocess(self, model_output):
    print("Postprocessing predictions...")
    final_diarisations = []
    
    for annotation in model_output:
        file_timeline = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            file_timeline.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker
            })
        final_diarisations.append(file_timeline)
        
    return final_diarisations