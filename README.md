# Easy localhost mode

For real upload + model inference, run this from the project root:

```bat
start-real-ai-localhost.cmd
```

The first run installs TensorFlow, Pillow, and NumPy into `.real-ai`.

For UI-only server mode:

```bat
start-localhost.cmd
```

Both start a local Node.js server at `http://localhost:3000`.

Real analysis uses `POST /api/analyze`, saves the uploaded image in `localhost/uploads`, runs `best_model.keras`, and returns pneumonia probability plus Grad-CAM when TensorFlow is available.
