#Import Relevant Modules
from msclap import CLAP
from openai import OpenAI
import torch
from huggingface_hub import snapshot_download
import torch
# from modelscope.pipelines import pipeline as pl
# from modelscope.outputs import OutputKeys
# import pathlib
from moviepy.editor import *
from transformers import pipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from diffusers.utils import export_to_gif
from diffusers import AnimateDiffPipeline, LCMScheduler,DDIMScheduler, MotionAdapter
# Initialize OpenAI client
llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

#Audio captioning function using MSCLAP(Not using for the project)
def audio_caption(file_loc):
# Load and initialize CLAP
    clap_model = CLAP(version = 'clapcap', use_cuda=False)

#Load audio files
    audio_files = [file_loc]

# Generate captions for the recording
    captions = clap_model.generate_caption(audio_files, resample=True, beam_size=5, entry_length=67, temperature=0.01)

# Print the result

    print(f"Audio file: {audio_files} \n")
    print(f"Generated caption: {captions} \n")
    return captions

#Video Prompt generation using a local llm by utilizing output recognized class from classification
def generate_prompt(audio_caption):
    # Check if inputs are empty
    
    # Define the prompt
    prompt = f"Given the audio caption: '{audio_caption}', generate a prompt description of a video that matches this audio.The prompt should also mention the necessary camera details and style of the video. The prompt should be within 75 tokens. **Prompt Start:**"

    response = llm.chat.completions.create(
        model="local-model", 
        messages=[
            {"role": "system", "content": "You are a prompt generator for a text to video synthesizer model.Give relevant video prompts."},
            {"role": "user", "content": prompt},
        ],
    )

     # Extract the stylized text from the response
    output = response.choices[0].message.content
    start = output.find("**Prompt Start:**")
    end = output.find("**Prompt End")

# Extract the prompt
    prompt = output[0:end].strip()
    print("The generated video prompt: ", prompt)


    return prompt

def audvid_sync(audio_file,video_file):
    

# Load gif and audio file
    gif = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)

    # Make sure the gif and audio have the same duration
    gif = gif.set_duration(audio.duration)

    # Convert gif to mp4
    gif.write_videofile("temp.mp4", codec='mpeg4')

    # Load the converted gif (now an mp4 video)
    video = VideoFileClip("temp.mp4")

    # Set the audio of the video
    final_video = video.set_audio(audio)
    video_path = f"{audio_file[:-4]}.mp4"
    # Write the result to a file
    final_video.write_videofile(video_path, codec='mpeg4')



#Basic video gen model(not using for project, just for quick testing)
def gen_video(prompt,path):


    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    vprompt = prompt
    video_frames = pipe(vprompt, num_inference_steps=25).frames
    video_frames = video_frames[0]
    video_path = f"{path[:-4]}.mp4"
    export_to_video(video_frames,video_path)
#Animation model.
def LCM_Model(vprompt,path):
    

    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=vprompt,
        negative_prompt="bad quality, worse quality, low resolution",
        num_frames=16,
        guidance_scale=2.0,
        num_inference_steps=6,
        generator=torch.Generator("cpu").manual_seed(0),
    )
    frames = output.frames[0]
    video_path = f"{path[:-4]}.gif"
    export_to_gif(frames, video_path)
    return video_path
#A better model, not using for the project cuz its not working rn
def zeroscope_gen_video(vprompt,file_path):

    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt = vprompt
    video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    video_path = export_to_video(video_frames)
    # Save the video locally
    local_path = f"{file_path[:-4]}.mp4"
    video_frames.save(local_path)

    print(f"Video saved locally at: {local_path}")
    # video_path = f"{file_path[:-4]}.mp4"
    # export_to_video(video_frames,video_path)
#alternative decent video model option, doesnt work , so not using it
def damomodel(vprompt, file_path):

    model_dir = pathlib.Path('weights')
    snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis',
                    repo_type='model', local_dir=model_dir)

    pipe = pl('text-to-video-synthesis', model_dir.as_posix())
    test_text = {
            'text': 'A panda eating bamboo on a rock.',
        }
    output_video_path = pipe(test_text,)[OutputKeys.OUTPUT_VIDEO]
    print('output_video_path:', output_video_path)
#Our Audio Classification model
def audio_Classification(path):
    pipe = pipeline("audio-classification", model="shreyahegde/ast-finetuned-audioset-10-10-0.450_ESC50")
    predicted_class=pipe(path)[0]['label']
    print("The predicted audio caption is : ", predicted_class)
    return predicted_class
#Lora model, using for testing, very good output, but in gif form, hence long term not usable for project
def lora_model(Vprompt,file_path):

    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=(
            Vprompt
        ),
        negative_prompt="bad quality, worse quality",
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=torch.Generator("cpu").manual_seed(42),
    )
    frames = output.frames[0]
    local_path = f"{file_path[:-4]}.gif"
    export_to_gif(frames, local_path)

#Main function for audio to video.
def audio_2_video(file_path):
    caption = audio_Classification(file_path)
    # caption=audio_caption(file_path)
    prompt=generate_prompt(caption)
    #damomodel(prompt,file_path)
    # lora_model(prompt,file_path)
    video_file=LCM_Model(prompt,file_path)
    audvid_sync(file_path,video_file)
    #zeroscope_gen_video(prompt,file_path)
    
audio_2_video('1-13613-A-37.wav')        
#  # Call the modified function
# damomodel('Spiderman doing a backflip', 'Baby Crying Sound Effect (320 kbps).mp3')
