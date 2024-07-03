#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pathlib import Path
def take_photos(n=100,name="mradul"):
    import cv2
    import time
    from pathlib import Path
    image_dir=Path(f"data/train/{name}")
    if image_dir.is_dir():
        print("Photos folder is present")
    else:
        print("photos folder not found so creating one")
        image_dir.mkdir(parents=True)

    i=1
    cap=cv2.VideoCapture(0)
    while True:
        time.sleep(0.1)
        res,img=cap.read()
        print(f"Taking image{i}")
        cv2.imwrite( f"data/train/{name}/image{i}.jpg",img)
        print(f"Saving image{i}")
        i+=1
        if i==n+1:
            i=1
            image_dir=Path(f"data/test/{name}")
            if image_dir.is_dir():
                print("Photos folder is present")
            else:
                print("photos folder not found so creating one")
                image_dir.mkdir(parents=True)
            while True:
                time.sleep(0.1)
                res,img=cap.read()
                print(f"Taking image{i}")
                cv2.imwrite(f"data/test/{name}/image{i}.jpg",img)
                print(f"Saving image{i}")
                i+=1
                if i==(0.25*n)+1:
                    break
            cap.release()
            cv2.destroyAllWindows()
            break
    
def clean_photos(location):
    for file in location.glob("*"):
        file.unlink()
from timeit import default_timer as timer
start_time=timer()
if __name__=="__main__":
    take_photos(100)
end_time=timer()
print(end_time-start_time)


# In[21]:





# In[ ]:




