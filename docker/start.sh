docker build -t piro1_136280_136236 ../src ../pickled_objects ../requirements.txt ../data
docker build -t piro1_136280_136236 - < piro1_136280_136236.src.tar.bz2
docker run --network none --mount type=bind,source="$(pwd)"/src,target=/app/src,readonly piro1_136280_136236 /app/src/data 6