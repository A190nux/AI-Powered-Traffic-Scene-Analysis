# AI-Powered Traffic Scene Analysis

This Google Colab notebook presents an AI-powered application for comprehensive traffic scene analysis. Leveraging Vision LLMs, it provides detailed descriptions of traffic environments, accurately detects license plates within the scene using Yolo11, and extracts the alphanumeric characters from them using Paddle. This tool is designed to assist in traffic monitoring, urban planning, and potentially law enforcement by offering automated insights into real-world traffic conditions.

The application integrates three powerful AI components:

  * **LLaVA (Large Language and Vision Assistant)**: Utilized for generating rich, contextual descriptions of the overall traffic scene. It focuses on aspects crucial for traffic management, such as road conditions, traffic flow, and unusual events.**bold text**
  * **YOLOv11 (You Only Look Once, version 11)**: A highly efficient object detection model specifically fine-tuned for robust license plate detection in diverse traffic scenarios.
  * **PaddleOCR**: A powerful Optical Character Recognition (OCR) engine used to accurately extract text from the detected license plates, even under varying conditions.

The user-friendly interface is built with Gradio, allowing for easy image uploads and real-time display of analysis results, including the scene description, the number of detected license plates, and a detailed JSON output of each detected plate's bounding box, extracted text, and confidence score.

## 1\. Setup and Dependency Instructions

To run this notebook in Google Colab, follow these steps:

1.  **Open in Google Colab**: Go to `File > Upload notebook` and select this `.ipynb` file, or if it's already on GitHub, `File > Open notebook > GitHub` and paste the URL.

2.  **GPU Runtime**: Ensure you are using a GPU runtime for optimal performance. Go to `Runtime > Change runtime type` and select `T4 GPU` or a similar GPU accelerator.

3.  **Install Dependencies**: Execute the following cell to install all necessary Python packages. This might take a few minutes.

    ```python
    !pip install ultralytics
    !pip install transformers_stream_generator
    !pip install -U bitsandbytes
    !pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    !pip install paddleocr
    !pip install -q ultralytics
    !pip install -q accelerate bitsandbytes
    !pip install -q gradio
    !wget -q -O license-plate-finetune-v1x.pt https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1x.pt
    !git clone https://github.com/A190nux/AI-Powered-Traffic-Scene-Analysis
    ```

4.  **Import Libraries**: Run the cell to import all the required libraries

    ```python
    import torch
    import gradio as gr
    from PIL import Image
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import json
    import requests
    import numpy as np
    import cv2
    ```


5.  **Load Models**: Run the cell containing the model loading code. This will download and initialize the LLaVA, YOLOv11, and PaddleOCR models. This step can also take some time depending on your internet connection.

    ```python
    # Load VLLM (LLaVA) for scene description
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=True,
    )
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Load Detection Model (YOLOv11)
    yolo_model = YOLO('/content/license-plate-finetune-v1x.pt')

    # Load OCR Model (PaddleOCR)
    ocr_model = PaddleOCR(use_textline_orientation=True)

    print("Models loaded successfully!")
    ```

6.  **Define `process_traffic_image` function**: Execute the cell containing the `process_traffic_image` function definition.

7.  **Launch Gradio App**: Finally, run the cell that sets up and launches the Gradio interface. A public URL will be provided, which you can open in your browser to interact with the application.

    ```python
    # Gradio App Code (as provided in your .py file)
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚦 AI-Powered Traffic Scene Analysis")
        gr.Markdown(
            "Upload an image of a traffic scene. The application will generate a detailed description of the scene, "
            "detect license plates, and extract the text from them."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Traffic Scene Image")

                gr.Markdown("### ⚙️ Adjustable Parameters")
                with gr.Accordion("Fine-tune model settings", open=False):
                    temp = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="VLLM Temperature")
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="VLLM Top_p")
                    yolo_confidence = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="YOLO Confidence Threshold")
                    ocr_confidence = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="OCR Confidence Threshold")

                submit_btn = gr.Button("Analyze Image", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📝 Analysis Results")
                output_description = gr.Textbox(label="Scene Description", lines=5)
                output_plate_count = gr.Number(label="Number of Plates Detected")
                output_json = gr.JSON(label="Combined JSON Output")

        submit_btn.click(
            fn=process_traffic_image,
            inputs=[input_image, yolo_confidence, ocr_confidence, temp, top_p],
            outputs=[output_description, output_plate_count, output_json]
        )

        gr.Examples(
            examples=[
                [
                    "/content/AI-Powered-Traffic-Scene-Analysis/images/Example_Image.jpg",
                    0.25,
                    0.25,
                    0.3,
                    0.3
                ]
            ],
            inputs=[input_image, yolo_confidence, ocr_confidence, temp, top_p],
            outputs=[output_description, output_plate_count, output_json],
            fn=process_traffic_image,
            cache_examples=True,
            examples_per_page=1
        )

    demo.launch(debug=True, share=True)
    ```

## 2\. Prompt Engineering Rationale and Parameter Choices

### Prompt Engineering for LLaVA Scene Description

The prompt for the LLaVA model is carefully crafted to elicit specific and useful information for traffic monitoring purposes, while explicitly excluding irrelevant details.

  * **Focus Areas**: The prompt directs LLaVA to concentrate on:
      * **Road condition**: Presence of cracks, potholes, etc.
      * **Traffic density**: Busyness in terms of people and vehicles.
      * **Traffic flow**: Is it normal, congested, or problematic? State of traffic lights.
      * **Unique events**: Accidents, congregations of people.

    These are the expected points of interest in the image. We tell the LLM to focus on them because otherwise it will keep discussion the weather, atmpsphere, etc.

  * **Exclusions**: The prompt explicitly tells LLaVA *not* to:
      * Mention the "general atmosphere."
      * Count vehicles or pedestrians.
      * Focus on buildings.

    We execlude these explicitly because the LLM keeps bringing them up and making up random numbers for the counts.

  * **Purpose-driven description**: The request for a "concise and factual description of the state of the traffic... reliable for monitoring the area and send personnel to interfere if necessary" guides the model to provide actionable insights.

    We should also mention the reason why we are doing this to help the LLM focus on the important aspects. If the "why" is included in the prompt, it tends to produce better results.

### Parameter Choices

  * **VLLM Temperature (`temperature`)**:
      * **Range**: 0.1 to 1.0
      * **Default**: 0.3
      * **Rationale**: Lower temperatures (like 0.3) make the output more deterministic and factual, which is desirable for a reliable traffic monitoring system. Higher temperatures would introduce more randomness and creativity, making the model make up stories sometimes. If its less than 0.3, the LLM will try to replicate the data / format it was trained on even if it means ignoring the instructions.
  * **VLLM Top\_p (`top_p`)**:
      * **Range**: 0.1 to 1.0
      * **Default**: 0.3
      * **Rationale**: Similar to temperature, a high p value will result in the LLM repeating its sentences using slightly different words. If its lower than that, it will not be descriptive enough to be useful
  * **YOLO Confidence Threshold (`yolo_confidence`)**:
      * **Range**: 0.1 to 0.9
      * **Default**: 0.25
      * **Rationale**: This parameter determines the minimum confidence score for a detected object (license plate) to be considered valid. A default of 0.25 strikes a balance between detecting most plates and filtering out low-confidence, potentially false positives. At this value, it is able to detect the plates consistently, even if the image is slightly blurry. It could require adjustment depending on the expected speed limit of the road to acount for this.
  * **OCR Confidence Threshold (`ocr_confidence`)**:
      * **Range**: 0.1 to 0.9
      * **Default**: 0.25
      * **Rationale**: This threshold filters the recognized text from the OCR model. Only text segments with a confidence score above this value are included in the final extracted license plate text. Using this value is for the exact same reason as yolo confidence.

-----

## 3\. Sample Outputs

Here you will place examples of outputs from the Gradio application. For each example, you should provide:

1.  **Input Image:** A visual representation of the traffic scene (e.g., a screenshot of the image used).
2.  **Scene Description Text:** The `output_description` generated by LLaVA.
3.  **Number of Plates Detected:** The `output_plate_count` from the detection.
4.  **Combined JSON Output:** The full `output_json` showing the scene description and detailed information for each detected plate.

**Example 1:**

**Input Image:**
![image](https://github.com/user-attachments/assets/265d4772-dbb7-4d23-a19b-af8aae8208f0)


**Scene Description:**

```
The image depicts a busy city street with heavy traffic, including cars and bicycles. There are multiple traffic lights in the scene, some of which are green, indicating that the traffic is flowing. However, the traffic appears to be congested, with cars and bicycles waiting at the intersection.

There are several people on the street, some of whom are walking, while others are riding bicycles. The street is lined with buildings, and there is a notable presence of trees in the area.\n\nOverall, the traffic scene is quite busy, with a mix of vehicles and pedestrians, and the traffic appears to be congested, requiring attention from traffic authorities to manage the flow effectively."
```

**Number of Plates Detected:**
`3`

**Combined JSON Output:**

```json
{
  "scene_description": "The image depicts a busy city street with heavy traffic, including cars and bicycles. There are multiple traffic lights in the scene, some of which are green, indicating that the traffic is flowing. However, the traffic appears to be congested, with cars and bicycles waiting at the intersection.\n\nThere are several people on the street, some of whom are walking, while others are riding bicycles. The street is lined with buildings, and there is a notable presence of trees in the area.\n\nOverall, the traffic scene is quite busy, with a mix of vehicles and pedestrians, and the traffic appears to be congested, requiring attention from traffic authorities to manage the flow effectively.",
  "number_of_plates_detected": 3,
  "detected_plates": [
    {
      "bounding_box": [
        1987,
        1570,
        2127,
        1617
      ],
      "text": "EUG·1512",
      "confidence": 0.9152898788452148
    },
    {
      "bounding_box": [
        2623,
        1545,
        2786,
        1614
      ],
      "text": "BAP-5493",
      "confidence": 0.9740387201309204
    },
    {
      "bounding_box": [
        3173,
        1543,
        3269,
        1589
      ],
      "text": "GC1:4125",
      "confidence": 0.903364896774292
    }
  ]
}
```

## 4\. Test Images

Below are 2 test images with their corresponding scene descriptions. You will need to add the actual image files and manually provide the scene descriptions generated by the application for these images.

**Test Image 1:**

  * **Image File:** `images/image_test_1.jpg`
![image3](https://github.com/user-attachments/assets/1b7e779f-037e-4a6b-94f1-1ec9b4242eac)

  * **Scene Description:**
    ```
    The image depicts a busy city street with multiple cars and traffic lights. The traffic appears to be flowing normally, and there are no visible problems such as cracks or potholes on the road.
    The traffic lights are functioning properly, and there are no unique events like accidents or congregations of people. The street is bustling with activity, and the vehicles are moving in a typical urban setting.
    ```

**Test Image 2:**

  * **Image File:** `images/image_test_2.jpg`
![image4](https://github.com/user-attachments/assets/65c1f491-b189-4c61-bc94-1adc672a2e1c)

  * **Scene Description:**
    ```
    The image depicts a busy city street with multiple cars and traffic lights. There are at least six cars visible in the scene, with some positioned closer to the foreground and others further back.
    Traffic lights can be seen at various points along the street, indicating that the area is well-regulated for vehicle and pedestrian traffic.

    The street appears to be in good condition, with no visible cracks or potholes. However, the traffic seems to be moving at a moderate pace, with no signs of congestion or accidents.
    The overall state of the road is reliable for monitoring and ensuring the smooth flow of traffic
    ```
