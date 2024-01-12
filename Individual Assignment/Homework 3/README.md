Bước 1: Cài đặt các thư viện cần thiết

!pip install matplotlib==3.1.3

!pip install osmnet

!pip install folium

!pip install rtree

!pip install pygeos

!pip install geojson

!pip install geopandas

Bước 2: clone data từ https://github.com/CityScope/CSL_HCMC

Bước 3: Load ranh giới quận huyện và dân số quận huyện từ: Data\GIS\Population\population_HCMC\population_shapefile\Population_District_Level.shp

Bước 4: Load dữ liệu click của người dùng

Bước 5: Lọc ra 5 quận huyện có tốc độ tăng MẬT ĐỘ dân số nhanh nhất (Dùng dữ liệu 2019  và 2017)

Bước 6: Dùng spatial join (from geopandas.tools import sjoin) để lọc ra các điểm click của người dùng trong 5 quận/huyện hot nhất

Bước 7: chạy KMean cho top 5 quận huyện này. Lấy K = 20

Bước 8: Lưu 01 cụm điểm nhiều nhất trong các quận huyện ở Bước 5.

Bước 9: show lên bản đồ các cụm đông nhất theo từng quận huyện theo dạng HEATMAP

Bước 10: Lưu heatmap xuống file png

Lưu ý: Tải file notebook về  và nộp lên Moodel