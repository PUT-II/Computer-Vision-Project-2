docker run --network none --mount type=bind,source="$(pwd)",target=/app/src,readonly piro_ocr /app/src/data 6