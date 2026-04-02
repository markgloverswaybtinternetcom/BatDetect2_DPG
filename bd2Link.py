import os

bd2_link = os.path.join(os.getcwd(), "batdetect2")
if not os.path.exists(bd2_link):
    print("No link {bd2_link=}")
    # Create symbolic link to BatDetect2 code
    bd2 = os.path.join(os.getcwd(), "libs", "batdetect2", "batdetect2")
    os.symlink(bd2, bd2_link, target_is_directory=True)        
