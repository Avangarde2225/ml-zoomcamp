FROM public.ecr.aws/lambda/python:3.10

COPY model_2024_hairstyle_v2.tflite .

RUN pip install numpy==1.23.1

# Install the specific tflite runtime wheel without dependencies
RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install Pillow

# Copy your lambda function
COPY lambda_function.py .

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ] 