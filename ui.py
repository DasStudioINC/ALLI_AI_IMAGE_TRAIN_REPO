import os
import threading
import customtkinter as ctk
from PIL import Image as PILImage, ImageTk
from tkinter import messagebox

# Import your backend script (app.py)
import app

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class BachAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Bach AI image gen")
        self.geometry("1150x800")
        self.configure(fg_color="#050114")

        self.pipeline = None
        self.latest_entries = [] # Stores current batch results for GitHub push

        # Left Panel (Controls)
        self.left_frame = ctk.CTkFrame(self, fg_color="#0a0635", corner_radius=15, width=400)
        self.left_frame.pack(side="left", fill="y", padx=15, pady=15)
        self.left_frame.pack_propagate(False)

        # Right Main Container (Scrollable Feed + Bottom GitHub Button Frame)
        self.right_container = ctk.CTkFrame(self, fg_color="transparent")
        self.right_container.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        self.right_frame = ctk.CTkScrollableFrame(self.right_container, fg_color="#0a0635", corner_radius=15)
        self.right_frame.pack(side="top", fill="both", expand=True, pady=(0, 10))

        # Push to Github Button at the bottom of the right view
        self.github_btn = ctk.CTkButton(
            self.right_container,
            text="Push to Github",
            command=self.start_github_push_thread,
            fg_color="#1a1261", hover_color="#2c1f96",
            border_width=2, border_color="#3626b8",
            corner_radius=12, height=45,
            font=("Arial", 14, "bold"),
            state="disabled" # Disabled until a batch of images is generated
        )
        self.github_btn.pack(side="bottom", fill="x")

        self.build_left_controls()
        self.build_right_feed_placeholder()

        # Load model in background on startup (does NOT generate images, only loads weights into RAM/VRAM)
        threading.Thread(target=self.load_model_background, daemon=True).start()

    def build_left_controls(self):
        ctk.CTkLabel(self.left_frame, text="Bach AI image gen", font=("Arial", 22, "bold"), text_color="white").pack(anchor="w", padx=20, pady=(20, 10))
        ctk.CTkLabel(self.left_frame, text="Prompt", font=("Arial", 18, "bold"), text_color="white").pack(anchor="w", padx=20, pady=(10, 5))

        self.prompt_text = ctk.CTkTextbox(self.left_frame, fg_color="#100b45", text_color="white", corner_radius=10, height=140, font=("Arial", 13))
        self.prompt_text.pack(fill="x", padx=20, pady=5)
        self.prompt_text.insert("1.0", "Close-up portrait of Spider-Man with glowing white eyes, a cyan glowing spider chest logo, and dramatic red smoke on a dark black background.")

        img_row = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        img_row.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(img_row, text="#IMG", font=("Arial", 18, "bold"), text_color="white").pack(side="left")

        self.batch_slider_val = ctk.IntVar(value=2)
        ctk.CTkOptionMenu(
            img_row, 
            values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"], 
            variable=ctk.StringVar(value="2"),
            command=lambda val: self.batch_slider_val.set(int(val)),
            fg_color="#1a1261", button_color="#281c8a", button_hover_color="#3626b8",
            width=120
        ).pack(side="right")

        self.status_label = ctk.CTkLabel(self.left_frame, text="Status: Loading AI model into memory...", font=("Arial", 12), text_color="#a39ac4", wraplength=340, justify="left")
        self.status_label.pack(anchor="w", padx=20, pady=10)

        # GENERATE BUTTON: This is the ONLY trigger for image creation
        self.generate_btn = ctk.CTkButton(
            self.left_frame, 
            text="Button to generate", 
            command=self.start_generation_thread,
            fg_color="#1a1261", hover_color="#2c1f96",
            border_width=2, border_color="#3626b8",
            corner_radius=12, height=50,
            font=("Arial", 15, "bold")
        )
        self.generate_btn.pack(fill="x", padx=20, pady=(10, 20), side="bottom")

    def build_right_feed_placeholder(self):
        ctk.CTkLabel(self.right_frame, text="Generated images and reverse-engineered prompts will appear here.", text_color="#7367a1", font=("Arial", 14)).pack(pady=100)

    def load_model_background(self):
        try:
            # This only loads Stable Diffusion into GPU/CPU memory. It does NOT generate images.
            self.pipeline = app.initialize_generator()
            self.status_label.configure(text="Status: AI Ready! Enter prompt and click generate.")
        except Exception as e:
            self.status_label.configure(text=f"Status: Error loading model ({e})")

    def start_generation_thread(self):
        """Triggered strictly when the generate button is clicked."""
        if not self.pipeline:
            messagebox.showwarning("Loading", "The AI model is still loading into memory. Please wait a moment.")
            return
            
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a valid prompt.")
            return
            
        num_images = self.batch_slider_val.get()
        self.generate_btn.configure(state="disabled", text="Generating...")
        
        # Run batch generation and Ollama analysis safely in a background thread
        threading.Thread(target=self.run_generation_task, args=(prompt, num_images), daemon=True).start()

    def run_generation_task(self, prompt, num_images):
        try:
            self.after(0, lambda: self.status_label.configure(text=f"Status: Generating {num_images} images..."))
            saved_images = app.batch_generate_images(self.pipeline, prompt, num_images=num_images)
            
            self.after(0, lambda: self.status_label.configure(text="Status: Analyzing images with Ollama..."))
            self.latest_entries = app.reverse_engineer_prompts(saved_images)
            
            app.save_to_json(self.latest_entries, app.JSON_OUTPUT_PATH)
            
            self.after(0, lambda: self.populate_results_feed(self.latest_entries))
            self.after(0, lambda: self.status_label.configure(text=f"Status: Done! Ready to push to GitHub."))
            self.after(0, lambda: self.github_btn.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {e}"))
        finally:
            self.after(0, lambda: self.generate_btn.configure(state="normal", text="Button to generate"))

    def start_github_push_thread(self):
        if not self.latest_entries:
            messagebox.showwarning("Warning", "No generated dataset available to push.")
            return
            
        self.github_btn.configure(state="disabled", text="Pushing to GitHub...")
        threading.Thread(target=self.run_github_push_task, daemon=True).start()

    def run_github_push_task(self):
        def update_status(msg):
            self.after(0, lambda: self.status_label.configure(text=f"Status: {msg}"))

        success = app.push_to_github(self.latest_entries, status_callback=update_status)
        
        def reset_btn():
            self.github_btn.configure(state="normal", text="Push to Github")
            if success:
                messagebox.showinfo("Success", "Successfully pushed dataset updates and images to GitHub!")
            else:
                messagebox.showerror("Error", "Failed to push to GitHub. Check terminal logs.")

        self.after(0, reset_btn)

    def populate_results_feed(self, entries):
        for widget in self.right_frame.winfo_children():
            widget.destroy()

        for entry in entries:
            card = ctk.CTkFrame(self.right_frame, fg_color="#100b45", corner_radius=12, border_width=2, border_color="#22176b")
            card.pack(fill="x", padx=10, pady=10, ipady=10)

            filename = entry["url"].split("/")[-1]
            local_img_file = os.path.join(app.OUTPUT_DIR, filename)
            
            if os.path.exists(local_img_file):
                raw_img = PILImage.open(local_img_file)
                raw_img = raw_img.resize((220, 220))
                photo = ImageTk.PhotoImage(raw_img)
                
                img_lbl = ctk.CTkLabel(card, image=photo, text="")
                img_lbl.image = photo
                img_lbl.pack(side="left", padx=15, pady=15)
            else:
                ctk.CTkLabel(card, text="Image Missing", width=220, height=220, fg_color="#0a0635", text_color="white").pack(side="left", padx=15, pady=15)

            prompt_container = ctk.CTkFrame(card, fg_color="transparent")
            prompt_container.pack(side="right", fill="both", expand=True, padx=15, pady=15)

            ctk.CTkLabel(prompt_container, text="Reversed Engineered Prompt:", font=("Arial", 13, "bold"), text_color="#b1a6e0").pack(anchor="w")

            prompt_box = ctk.CTkTextbox(prompt_container, fg_color="#0a0635", text_color="white", height=120, font=("Arial", 12), wrap="word")
            prompt_box.pack(fill="x", pady=5)
            prompt_box.insert("1.0", entry["prompt"])
            prompt_box.configure(state="disabled")

if __name__ == "__main__":
    app_window = BachAIApp()
    app_window.mainloop()