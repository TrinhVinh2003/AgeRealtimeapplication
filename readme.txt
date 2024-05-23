#tải model từ git ,chạy tren terminal
git clone https://github.com/TrinhVinh2003/AgeRealtimeapplication.git

# tạo virirtual enviroment
python -m venv venv

# kích hoạt venv :
venv/Scripts\activate.bat  (window)

source venv/bin/activate.bat  (Mac/Linux)

# tải thư viện framework flask để chạy được trên web
pip install flask

# tải các thư viện để chạy model (tensorflow , cvlib, numpy,pandas, seaborn , matplotplit, ..)
pip install -r setup.txt


# chạy application:
flask run