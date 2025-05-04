import gradio as gr
import time
import datetime
import random
import os
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import base64
import io

from modules.video_queue import JobStatus, Job
from modules.prompt_handler import get_section_boundaries, get_quick_prompts, parse_timestamped_prompt
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html


def create_interface(
    process_fn, 
    monitor_fn, 
    end_process_fn, 
    update_queue_status_fn,
    load_lora_file_fn, 
    job_queue,
    settings,
    default_prompt: str = '"[1s: The person waves hello] [3s: The person jumps up and down] [5s: The person does a dance]',
    lora_names: list = [],
    lora_values: list = []
):
    """
    Create the Gradio interface for the video generation application
    
    Args:
        process_fn: Function to process a new job
        monitor_fn: Function to monitor an existing job
        end_process_fn: Function to cancel the current job
        update_queue_status_fn: Function to update the queue status display
        default_prompt: Default prompt text
        lora_names: List of loaded LoRA names
        
    Returns:
        Gradio Blocks interface
    """
    # Get section boundaries and quick prompts
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()
    
    # Create the interface
    css = make_progress_bar_css()
    css += """
    .contain-image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
        background: #222;
    }
    """

    css += """
    #fixed-toolbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 1000;
        background: rgb(11, 15, 25);
        color: #fff;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-bottom: 1px solid #4f46e5;
    }
    #toolbar-add-to-queue-btn button {
        font-size: 14px !important;
        padding: 4px 16px !important;
        height: 32px !important;
        min-width: 80px !important;
    }



    .gr-button-primary{
        color:white;
    }
    body, .gradio-container {
        padding-top: 40px !important; 
    }
    """

    css += """
    .narrow-button {
        min-width: 40px !important;
        width: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    """

    block = gr.Blocks(css=css, title="FramePack Studio", theme="soft").queue()
    
    with block:

        with gr.Row(elem_id="fixed-toolbar"):
            gr.Markdown("<h1 style='margin:0;color:white;'>FramePack Studio</h1>")
            with gr.Column(scale=1):
                queue_stats_display = gr.Markdown("<p style='margin:0;color:white;'>Queue: 0 | Completed: 0</p>")
            with gr.Column(scale=0):
                refresh_stats_btn = gr.Button("⟳", elem_id="refresh-stats-btn")
        
        
        # Hidden state to track the selected model type
        selected_model_type = gr.State("Original")
        
        with gr.Tabs():
            with gr.Tab("Generate (Original)", id="original_tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        input_image = gr.Image(
                            sources='upload',
                            type="numpy",
                            label="Image (optional)",
                            height=420,
                            elem_classes="contain-image"
                        )

                        with gr.Accordion("Latent Image Options", open=False):
                            latent_type = gr.Dropdown(
                                ["Black", "White", "Noise", "Green Screen"], label="Latent Image", value="Black", info="Used as a starting point if no image is provided"
                            )
                        
                        prompt = gr.Textbox(label="Prompt", value=default_prompt)

                        with gr.Accordion("Prompt Parameters", open=False):
                            blend_sections = gr.Slider(
                                minimum=0, maximum=10, value=4, step=1,
                                label="Number of sections to blend between prompts"
                            )
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                                total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                            with gr.Row("LoRAs"):
                                lora_selector = gr.Dropdown(
                                    choices=lora_names,
                                    label="Select LoRAs to Load",
                                    multiselect=True,
                                    value=[],
                                    info="Select one or more LoRAs to use for this job"
                                )

                                lora_sliders = {}
                                for lora in lora_names:
                                    lora_sliders[lora] = gr.Slider(
                                        minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                        label=f"{lora} Weight", visible=False, interactive=True
                                    )

                            with gr.Row("Metadata"):
                                json_upload = gr.File(
                                    label="Upload Metadata JSON (optional)",
                                    file_types=[".json"],
                                    type="filepath",
                                    height=100,
                                )
                                save_metadata = gr.Checkbox(label="Save Metadata", value=True, info="Save to JSON file")   
                            with gr.Row("TeaCache"):
                                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                            
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=31337, precision=0)
                                randomize_seed = gr.Checkbox(label="Randomize", value=False, info="Generate a new random seed for each job")
                            

                        with gr.Accordion("Advanced Parameters", open=False):    
                            latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')  # Should not change
                            cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                            gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                            rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change
                            gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                        with gr.Accordion("Output Parameters", open=False): 
                            mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                            clean_up_videos = gr.Checkbox(
                                label="Clean up video files",
                                value=True,
                                info="If checked, only the final video will be kept after generation."
                            )
                            
                    with gr.Column():
                        preview_image = gr.Image(label="Next Latents", height=150, visible=True, type="numpy", interactive=False)
                        result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')

                        with gr.Row():  
                            current_job_id = gr.Textbox(label="Current Job ID", visible=True, interactive=True) 
                            end_button = gr.Button(value="Cancel Current Job", interactive=True) 
                            start_button = gr.Button(value="Add to Queue", elem_id="toolbar-add-to-queue-btn")

            with gr.Tab("Generate (F1)", id="f1_tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        f1_input_image = gr.Image(
                            sources='upload',
                            type="numpy",
                            label="Image (optional)",
                            height=420,
                            elem_classes="contain-image"
                        )

                        with gr.Accordion("Latent Image Options", open=False):
                            f1_latent_type = gr.Dropdown(
                                ["Black", "White", "Noise", "Green Screen"], label="Latent Image", value="Black", info="Used as a starting point if no image is provided"
                            )
                        
                        f1_prompt = gr.Textbox(label="Prompt", value=default_prompt)

                        with gr.Accordion("Prompt Parameters", open=False):
                            f1_blend_sections = gr.Slider(
                                minimum=0, maximum=10, value=4, step=1,
                                label="Number of sections to blend between prompts"
                            )
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                f1_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                                f1_total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                            with gr.Row("LoRAs"):
                                f1_lora_selector = gr.Dropdown(
                                    choices=lora_names,
                                    label="Select LoRAs to Load",
                                    multiselect=True,
                                    value=[],
                                    info="Select one or more LoRAs to use for this job"
                                )

                                f1_lora_sliders = {}
                                for lora in lora_names:
                                    f1_lora_sliders[lora] = gr.Slider(
                                        minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                        label=f"{lora} Weight", visible=False, interactive=True
                                    )

                            with gr.Row("Metadata"):
                                f1_json_upload = gr.File(
                                    label="Upload Metadata JSON (optional)",
                                    file_types=[".json"],
                                    type="filepath",
                                    height=100,
                                )
                                f1_save_metadata = gr.Checkbox(label="Save Metadata", value=True, info="Save to JSON file")   
                            with gr.Row("TeaCache"):
                                f1_use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                                f1_n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                            
                            with gr.Row():
                                f1_seed = gr.Number(label="Seed", value=31337, precision=0)
                                f1_randomize_seed = gr.Checkbox(label="Randomize", value=False, info="Generate a new random seed for each job")

                        with gr.Accordion("Advanced Parameters", open=False):    
                            f1_latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')
                            f1_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                            f1_gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                            f1_rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                            f1_gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                        with gr.Accordion("Output Parameters", open=False): 
                            f1_mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                            f1_clean_up_videos = gr.Checkbox(
                                label="Clean up video files",
                                value=True,
                                info="If checked, only the final video will be kept after generation."
                            )
                            
                    with gr.Column():
                        f1_preview_image = gr.Image(label="Next Latents", height=150, visible=True, type="numpy", interactive=False)
                        f1_result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                        f1_progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        f1_progress_bar = gr.HTML('', elem_classes='no-generating-animation')

                        with gr.Row():  
                            f1_current_job_id = gr.Textbox(label="Current Job ID", visible=True, interactive=True) 
                            f1_end_button = gr.Button(value="Cancel Current Job", interactive=True) 
                            f1_start_button = gr.Button(value="Add to Queue", elem_id="toolbar-add-to-queue-btn")

            with gr.Tab("Queue"):
                with gr.Row():
                    with gr.Column():
                        # Create a container for the queue status
                        with gr.Row():
                            queue_status = gr.DataFrame(
                                headers=["Job ID", "Type", "Status", "Created", "Started", "Completed", "Elapsed", "Preview"],
                                datatype=["str", "str", "str", "str", "str", "str", "str", "image"],
                                label="Job Queue"
                            )
                        with gr.Row():
                            refresh_button = gr.Button("Refresh Queue")
                            refresh_button.click(
                                fn=update_queue_status_fn,
                                inputs=[],
                                outputs=[queue_status]
                            )
                        # Create a container for thumbnails
                        with gr.Row():
                            thumbnail_container = gr.Column()
                            thumbnail_container.elem_classes = ["thumbnail-container"]

                        # Add CSS for thumbnails
                        css += """
                        .thumbnail-container {
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                            padding: 10px;
                        }
                        .thumbnail-item {
                            width: 100px;
                            height: 100px;
                            border: 1px solid #444;
                            border-radius: 4px;
                            overflow: hidden;
                        }
                        .thumbnail-item img {
                            width: 100%;
                            height: 100%;
                            object-fit: cover;
                        }
                        """

            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            value=settings.get("output_dir"),
                            placeholder="Path to save generated videos"
                        )
                        metadata_dir = gr.Textbox(
                            label="Metadata Directory",
                            value=settings.get("metadata_dir"),
                            placeholder="Path to save metadata files"
                        )
                        lora_dir = gr.Textbox(
                            label="LoRA Directory",
                            value=settings.get("lora_dir"),
                            placeholder="Path to LoRA models"
                        )
                        auto_save = gr.Checkbox(
                            label="Auto-save settings",
                            value=settings.get("auto_save_settings", True)
                        )
                        save_btn = gr.Button("Save Settings")
                        status = gr.HTML("")

                        def save_settings(output_dir, metadata_dir, lora_dir, auto_save):
                            try:
                                settings.update({
                                    "output_dir": output_dir,
                                    "metadata_dir": metadata_dir,
                                    "lora_dir": lora_dir,
                                    "auto_save_settings": auto_save
                                })
                                return "<p style='color:green;'>Settings saved successfully!</p>"
                            except Exception as e:
                                return f"<p style='color:red;'>Error saving settings: {str(e)}</p>"

                        save_btn.click(
                            fn=save_settings,
                            inputs=[output_dir, metadata_dir, lora_dir, auto_save],
                            outputs=[status]
                        )

        # Connect the main process function
        def process_with_queue_update(*args):
            # Extract all arguments
            input_image, prompt_text, n_prompt, seed_value, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, save_metadata_checked, blend_sections, latent_type, *lora_args = args
            
            # Call the process function with all arguments
            result = process_fn(input_image, prompt_text, n_prompt, seed_value, total_second_length, 
                latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                use_teacache, mp4_crf, save_metadata_checked, blend_sections, latent_type, *lora_args)
            
            # If a job ID was created, automatically start monitoring it and update queue
            if result and result[1]:  # Check if job_id exists in results
                job_id = result[1]
                queue_status_data = update_queue_status_fn()
                return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data]
            
            return result + [update_queue_status_fn()]

        # Connect the buttons to their respective functions
        start_button.click(
            # Pass "Original" model type
            fn=lambda *args: process_with_queue_update("Original", *args),
            inputs=[
                input_image, prompt, n_prompt, seed, total_second_length,
                latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                use_teacache, mp4_crf, save_metadata, blend_sections, latent_type,
                clean_up_videos, lora_selector
            ],
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, seed]
        )

        end_button.click(
            fn=end_process_fn,
            inputs=[],
            outputs=[queue_status]
        )

        current_job_id.change(
            fn=monitor_fn,
            inputs=[current_job_id],
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button]
        )
        
        refresh_stats_btn.click(
            fn=lambda: (get_queue_stats(), update_queue_status_fn()),
            inputs=None,
            outputs=[queue_stats_display, queue_status]
        )

        # Function to get queue statistics
        def get_queue_stats():
            try:
                # Get all jobs from the queue
                jobs = job_queue.get_all_jobs()
                
                # Count jobs by status
                status_counts = {
                    "QUEUED": 0,
                    "RUNNING": 0,
                    "COMPLETED": 0,
                    "FAILED": 0,
                    "CANCELLED": 0
                }
                
                for job in jobs:
                    if hasattr(job, 'status'):
                        status = str(job.status)
                        if status in status_counts:
                            status_counts[status] += 1
                
                # Format the display text
                stats_text = f"Queue: {status_counts['QUEUED']} | Running: {status_counts['RUNNING']} | Completed: {status_counts['COMPLETED']} | Failed: {status_counts['FAILED']} | Cancelled: {status_counts['CANCELLED']}"
                
                return f"<p style='margin:0;color:white;'>{stats_text}</p>"
                
            except Exception as e:
                print(f"Error getting queue stats: {e}")
                return "<p style='margin:0;color:white;'>Error loading queue stats</p>"

        # Function to update slider visibility based on selection
        def update_lora_sliders(selected_loras):
            updates = []
            for lora in lora_names:
                updates.append(gr.update(visible=(lora in selected_loras)))
            return updates

        # Connect the dropdown to the sliders
        lora_selector.change(
            fn=update_lora_sliders,
            inputs=[lora_selector],
            outputs=[lora_sliders[lora] for lora in lora_names]
        )

        # Function to load metadata from JSON file
        def load_metadata_from_json(json_path):
            if not json_path:
                return [gr.update(), gr.update()] + [gr.update() for _ in lora_names]
            
            try:
                import json
                
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                
                prompt = metadata.get('prompt')
                seed = metadata.get('seed')
                
                # Check for LoRA values in metadata
                lora_values = metadata.get('lora_values', {})
                
                print(f"Loaded metadata from JSON: {json_path}")
                print(f"Prompt: {prompt}, Seed: {seed}")
                
                # Update the UI components
                updates = [
                    gr.update(value=prompt) if prompt else gr.update(),
                    gr.update(value=seed) if seed is not None else gr.update()
                ]
                
                # Update LoRA sliders if they exist in metadata
                for lora in lora_names:
                    if lora in lora_values:
                        updates.append(gr.update(value=lora_values[lora]))
                    else:
                        updates.append(gr.update())
                
                return updates
                
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return [gr.update(), gr.update()] + [gr.update() for _ in lora_names]

        # Connect JSON metadata loader
        json_upload.change(
            fn=load_metadata_from_json,
            inputs=[json_upload],
            outputs=[prompt, seed] + [lora_sliders[lora] for lora in lora_names]
        )

        def format_queue_status(jobs):
            """Format job data for display in the queue status table"""
            rows = []
            for job in jobs:
                created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
                started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
                completed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

                # Calculate elapsed time
                elapsed_time = ""
                if job.started_at:
                    if job.completed_at:
                        start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                        complete_datetime = datetime.datetime.fromtimestamp(job.completed_at)
                        elapsed_seconds = (complete_datetime - start_datetime).total_seconds()
                        elapsed_time = f"{elapsed_seconds:.2f}s"
                    else:
                        # For running jobs, calculate elapsed time from now
                        start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                        current_datetime = datetime.datetime.now()
                        elapsed_seconds = (current_datetime - start_datetime).total_seconds()
                        elapsed_time = f"{elapsed_seconds:.2f}s (running)"

                # Get generation type from job data
                generation_type = getattr(job, 'generation_type', 'Original')

                # Convert base64 thumbnail to PIL Image for Gradio
                thumbnail = None
                if job.thumbnail:
                    try:
                        # Extract base64 data from data URL
                        base64_data = job.thumbnail.split(',')[1]
                        # Convert base64 to bytes
                        image_bytes = base64.b64decode(base64_data)
                        # Convert bytes to PIL Image
                        thumbnail = Image.open(io.BytesIO(image_bytes))
                    except Exception as e:
                        print(f"Error converting thumbnail: {e}")

                rows.append([
                    job.id[:6] + '...',
                    generation_type,
                    job.status.value,
                    created,
                    started,
                    completed,
                    elapsed_time,
                    thumbnail
                ])
            return rows

        # Create the queue status update function
        def update_queue_status_with_thumbnails():
            jobs = job_queue.get_all_jobs()
            for job in jobs:
                if job.status == JobStatus.PENDING:
                    job.queue_position = job_queue.get_queue_position(job.id)
            
            if job_queue.current_job:
                job_queue.current_job.status = JobStatus.RUNNING
            
            return format_queue_status(jobs)

        # Connect the refresh button
        refresh_button.click(
            fn=update_queue_status_with_thumbnails,
            inputs=[],
            outputs=[queue_status]
        )

        # Function to update F1 LoRA sliders
        def update_f1_lora_sliders(selected_loras):
            updates = []
            for lora in lora_names:
                updates.append(gr.update(visible=(lora in selected_loras)))
            return updates

        # Connect the F1 dropdown to the F1 sliders
        f1_lora_selector.change(fn=update_f1_lora_sliders, inputs=[f1_lora_selector], outputs=[f1_lora_sliders[lora] for lora in lora_names])

        # Add a refresh timer that updates the queue status every 2 seconds
        refresh_timer = gr.Number(value=0, visible=False)
        
        def refresh_timer_fn():
            """Updates the timer value periodically to trigger queue refresh"""
            return int(time.time())

        # --- Inputs for Original Model ---
        ips = [
            input_image,
            prompt,
            n_prompt, 
            seed, 
            total_second_length, 
            latent_window_size,
            steps, 
            cfg, 
            gs, 
            rs, 
            gpu_memory_preservation, 
            use_teacache, 
            mp4_crf,
            randomize_seed, 
            save_metadata, 
            blend_sections, 
            latent_type, 
            clean_up_videos,
            lora_selector
        ]
        # Add LoRA sliders to the input list
        ips.extend([lora_sliders[lora] for lora in lora_names])
                
        # --- Inputs for F1 Model ---
        f1_ips = [
            f1_input_image,
            f1_prompt,
            f1_n_prompt,
            f1_seed,
            f1_total_second_length,
            f1_latent_window_size,
            f1_steps,
            f1_cfg,
            f1_gs,
            f1_rs,
            f1_gpu_memory_preservation,
            f1_use_teacache,
            f1_mp4_crf,
            f1_randomize_seed,
            f1_save_metadata,
            f1_blend_sections,
            f1_latent_type,
            f1_clean_up_videos,
            f1_lora_selector
        ]

        # Add F1 LoRA sliders to the input list
        f1_ips.extend([f1_lora_sliders[lora] for lora in lora_names])

        # Modified process function that updates the queue status after adding a job
        def process_with_queue_update(model_type, *args):
            # Extract all arguments
            input_image, prompt_text, n_prompt, seed_value, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, randomize_seed_checked, save_metadata_checked, blend_sections, latent_type, clean_up_videos, selected_loras, *lora_args = args
            
            # Parse the prompt with the correct generation type
            prompt_sections = parse_timestamped_prompt(prompt_text, total_second_length, latent_window_size, model_type)
            
            # Use the current seed value as is for this job
            # Call the process function with all arguments
            # Pass the model_type to the backend process function
            result = process_fn(model_type, input_image, prompt_sections, n_prompt, seed_value, total_second_length,
                            latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                            use_teacache, mp4_crf, save_metadata_checked, blend_sections, latent_type, clean_up_videos, selected_loras, *lora_args)
            
            # If randomize_seed is checked, generate a new random seed for the next job
            new_seed_value = None
            if randomize_seed_checked:
                new_seed_value = random.randint(0, 21474)
                print(f"Generated new seed for next job: {new_seed_value}")
            
            # If a job ID was created, automatically start monitoring it and update queue
            if result and result[1]:  # Check if job_id exists in results
                job_id = result[1]
                queue_status_data = update_queue_status_fn()
                
                # Add the new seed value to the results if randomize is checked
                if new_seed_value is not None:
                    return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data, new_seed_value]
                else:
                    return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data, gr.update()]
            
            # If no job ID was created, still return the new seed if randomize is checked
            if new_seed_value is not None:
                return result + [update_queue_status_fn(), new_seed_value]
            else:
                return result + [update_queue_status_fn(), gr.update()]
            
        # Custom end process function that ensures the queue is updated
        def end_process_with_update():
            queue_status_data = end_process_fn()
            # Make sure to return the queue status data
            return queue_status_data

        # Connect the buttons to their respective functions
        start_button.click(
            # Pass "Original" model type
            fn=lambda *args: process_with_queue_update("Original", *args),
            inputs=ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, seed]
        )

        f1_start_button.click(
            # Pass "F1" model type
            fn=lambda *args: process_with_queue_update("F1", *args),
            inputs=f1_ips,
            # Update F1 outputs and shared queue/job ID
            outputs=[f1_result_video, f1_current_job_id, f1_preview_image, f1_progress_desc, f1_progress_bar, f1_start_button, f1_end_button, queue_status, f1_seed]
        )

        # Connect the end button to cancel the current job and update the queue
        end_button.click(
            fn=end_process_with_update,
            outputs=[queue_status]
        )

        # Auto-monitor the current job when job_id changes
        # Monitor original tab
        current_job_id.change(
            fn=monitor_fn,
            inputs=[current_job_id],
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button]
        )

        # Monitor F1 tab (using the same monitor function for now, assuming job IDs are unique)
        f1_current_job_id.change(
            fn=monitor_fn,
            inputs=[f1_current_job_id],
            outputs=[f1_result_video, f1_current_job_id, f1_preview_image, f1_progress_desc, f1_progress_bar, f1_start_button, f1_end_button]
        )

        # Connect F1 end button
        f1_end_button.click(
            fn=end_process_with_update,
            outputs=[queue_status] # Update shared queue status display
        )

        # Set up auto-refresh for queue status
        refresh_timer.change(
            fn=update_queue_status_fn,
            outputs=[queue_status] # Update shared queue status display
        )

        # Connect F1 JSON metadata loader
        f1_json_upload.change(
            fn=load_metadata_from_json,
            inputs=[f1_json_upload],
            outputs=[f1_prompt, f1_seed] + lora_values
        )

    return block


def format_queue_status(jobs):
    """Format job data for display in the queue status table"""
    rows = []
    for job in jobs:
        created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
        started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
        completed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

        # Calculate elapsed time
        elapsed_time = ""
        if job.started_at:
            if job.completed_at:
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                complete_datetime = datetime.datetime.fromtimestamp(job.completed_at)
                elapsed_seconds = (complete_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s"
            else:
                # For running jobs, calculate elapsed time from now
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                current_datetime = datetime.datetime.now()
                elapsed_seconds = (current_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s (running)"

        # Get generation type from job data
        generation_type = getattr(job, 'generation_type', 'Original')

        # Convert base64 thumbnail to PIL Image for Gradio
        thumbnail = None
        if job.thumbnail:
            try:
                # Extract base64 data from data URL
                base64_data = job.thumbnail.split(',')[1]
                # Convert base64 to bytes
                image_bytes = base64.b64decode(base64_data)
                # Convert bytes to PIL Image
                thumbnail = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                print(f"Error converting thumbnail: {e}")

        rows.append([
            job.id[:6] + '...',
            generation_type,
            job.status.value,
            created,
            started,
            completed,
            elapsed_time,
            thumbnail
        ])
    return rows
