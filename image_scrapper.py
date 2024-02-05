import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os 
images = ['apple',
        'banana',
        'blackgram',
        'chickpea',
        'coconut',
        'coffee',
        'cotton',
        'grapes',
        'jute',
        'kidneybeans',
        'lentil',
        'maize',
        'mango',
        'mothbeans',
        'mungbean',
        'muskmelon',
        'orange',
        'papaya',
        'pigeonpeas',
        'pomegranate',
        'rice',
        'watermelon']
for img in images:
    response = requests.get(f"https://www.google.com/search?q={img}&sxsrf=AJOqlzUuff1RXi2mm8I_OqOwT9VjfIDL7w:1676996143273&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiq-qK7gaf9AhXUgVYBHYReAfYQ_AUoA3oECAEQBQ&biw=1920&bih=937&dpr=1#imgrc=1th7VhSesfMJ4M")
    soup = BeautifulSoup(response.content,"html.parser")
    image_tags = soup.find_all("img")
    image_tag = image_tags[1]
    image_url =  image_tag['src']
    image_data = requests.get(image_url).content
    save_dir = "dataset/image_scrap_stuff/"
    with open(os.path.join(save_dir,f"{images.index(img)}.jpeg") , "wb") as f :
                           f.write(image_data)