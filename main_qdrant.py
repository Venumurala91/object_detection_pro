# main_qdrant.py
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import contextlib
import traceback
import os
import gradio as gr

# Import the new Qdrant-based logic
from face_search_logic_qdrant import FaceSearchEngine, FACE_DATABASE_PATH

# --- 1. SETUP ---
search_engine = FaceSearchEngine() # for search


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events, including Qdrant connection."""
    print("Application startup...")
    try:
        search_engine.load_models()
        # Call the Qdrant-specific setup method
        search_engine.load_or_create_collection()
        print("Startup complete. Server is ready.")
        yield
    except Exception as e:
        print(f"--- FATAL STARTUP ERROR ---")
        print("Could not connect to Qdrant. Is the Docker server running and healthy?")
        traceback.print_exc()
    finally:
        # The Qdrant client has a .close() method for graceful shutdown
        search_engine.disconnect_from_qdrant()
        print("Application shutdown.")


app = FastAPI(
    title="Face Search with Qdrant & Docker",
    description="A scalable face search application using Qdrant as the backend with progress tracking.",
    version="6.0.0",
    lifespan=lifespan
)


# --- 2. GRADIO UI HANDLERS ---
# NOTE: These handlers do not need to change, as the interface to the
# FaceSearchEngine class has been kept consistent.

def search_face_handler(query_image):
    if query_image is None: return "Please upload an image.", None
    try:
        data = search_engine.search_person(query_image)
        status_msg = data.get("status", "No status from engine.")
        results = data.get("results", [])
        gallery_images = [res["image_path"] for res in results]
        if gallery_images:
            details = "\n".join([f"- {os.path.basename(res['image_path'])} (Distance: {res['distance']:.4f})" for res in results])
            status_msg += f"\n\nMatches found:\n{details}"
        return status_msg, gallery_images
    except Exception as e:
        traceback.print_exc()
        return f"An unexpected error occurred: {e}", None


def add_images_handler(progress=gr.Progress(track_tqdm=True)):
    """
    Handler for the Gradio button. Passes the progress tracker to the logic function.
    """
    try:
        add_status = search_engine.add_images_from_directory(progress=progress)
        status_msg = add_status.get("status", "No status message.")
        images_added = add_status.get("images_added", 0)
        faces_added = add_status.get("faces_added", 0)
        return f"Update Complete!\nStatus: {status_msg}\nNew Images Scanned: {images_added}\nNew Faces Indexed: {faces_added}"
    except Exception as e:
        traceback.print_exc()
        return f"An unexpected error occurred: {e}"


# --- 3. GRADIO UI DEFINITION ---
# NOTE: The UI definition does not need to change at all.

with gr.Blocks(theme=gr.themes.Soft(), title="Face Search Engine") as demo:
    gr.Markdown("# Face Search Engine (Powered by Qdrant)")
    gr.Markdown("An interactive UI for a scalable face search system.")
    with gr.Tabs():
        with gr.TabItem("Search for a Face"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload Image")
                    image_input = gr.Image(type="numpy", label="Upload or Drag Image",
                                           sources=["upload", "clipboard", "webcam"])
                    search_button = gr.Button("Search", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("### 2. Results")
                    status_output = gr.Textbox(label="Status", interactive=False, lines=4)
                    gallery_output = gr.Gallery(label="Matching Images", show_label=False, object_fit="contain",
                                                height="auto", columns=4)
        with gr.TabItem("Manage Database"):
            gr.Markdown("### Update Face Database\nClick to scan the `face_database` directory for new images.")
            add_button = gr.Button("Scan Directory and Update Qdrant", variant="secondary")
            add_status_output = gr.Textbox(label="Update Status", interactive=False, lines=3)

    # Connect UI components to the handler functions
    search_button.click(fn=search_face_handler, inputs=image_input, outputs=[status_output, gallery_output])
    add_button.click(fn=add_images_handler, inputs=None, outputs=add_status_output)


# --- 4. APP RUNNER ---
@app.get("/", include_in_schema=False)
async def read_root():
    return RedirectResponse(url="/ui")


app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    if not os.path.exists(FACE_DATABASE_PATH):
        os.makedirs(FACE_DATABASE_PATH, exist_ok=True)
        print(f"Created face database directory: {FACE_DATABASE_PATH}")

    print("--- Starting Qdrant-backed FastAPI server ---")
    print("Ensure Docker container for Qdrant is running before starting.")
    print(f"Gradio UI available at: http://localhost:8000/ui")
    print("---")
    uvicorn.run("main_qdrant:app", host="0.0.0.0", port=8000, reload=True)

# docker run -p 6333:6333 -p 6334:6334 -v "%cd%\qdrant_storage:/qdrant/storage" qdrant/qdrant