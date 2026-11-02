import os
import urllib.request

# Sample food images from Unsplash
food_images = {
    'pizza': [
        'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=200',
        'https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=200',
        'https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=200',
        'https://images.unsplash.com/photo-1593560708920-61dd98c46a4e?w=200',
        'https://images.unsplash.com/photo-1534308983496-4fabb1a015ee?w=200',
    ],
    'burger': [
        'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=200',
        'https://images.unsplash.com/photo-1586190848861-99aa4a171e90?w=200',
        'https://images.unsplash.com/photo-1550547660-d9450f859349?w=200',
        'https://images.unsplash.com/photo-1571091718767-18b5b1457add?w=200',
        'https://images.unsplash.com/photo-1553979459-d2229ba7433b?w=200',
    ],
    'salad': [
        'https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=200',
        'https://images.unsplash.com/photo-1540420773420-3366772f4999?w=200',
        'https://images.unsplash.com/photo-1546793665-c74683f339c1?w=200',
        'https://images.unsplash.com/photo-1547496502-affa22d38842?w=200',
        'https://images.unsplash.com/photo-1505253716362-afaea1d3d1af?w=200',
    ]
}

base_dirs = [
    r"c:\Users\athar\OneDrive\Desktop\PR\Food",
    os.path.expanduser("~/Downloads/Food")
]

for base_dir in base_dirs:
    for cat, urls in food_images.items():
        train_dir = os.path.join(base_dir, "train", cat)
        test_dir = os.path.join(base_dir, "test", cat)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        for i, url in enumerate(urls[:4]):
            dst = os.path.join(train_dir, f"{cat}_{i}.jpg")
            if not os.path.exists(dst):
                try:
                    urllib.request.urlretrieve(url, dst)
                except Exception as e:
                    print(f"Error {cat} {i}: {e}")
                    
        for i, url in enumerate(urls[4:]):
            dst = os.path.join(test_dir, f"{cat}_{i}.jpg")
            if not os.path.exists(dst):
                try:
                    urllib.request.urlretrieve(url, dst)
                except Exception as e:
                    print(f"Error test {cat} {i}: {e}")

print("Food dataset created in PR/Food and Downloads/Food successfully!")
