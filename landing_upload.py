from landinglens import LandingLens
import os

# make sure the config.ini is setup properly in your ~/.landinglens/config.ini like

####################################################################################
# [DEFAULT]
# default_profile = aztrial
#
# [aztrial]
# #org=530
# api_key = hz5n5mwbrnbkxvo8fjxi4qgr4thv4xs
# api_secret = 58dwefps2922e6nzdd3kxr8ssl1q7sfsjbwt6dqj5h3kqozkp7m9zqk0qbvevt
# project_id = 12203
####################################################################################

# Summarize before uploading
llens = LandingLens()
media_list = llens.media.ls()
print("Before:", media_list)

# upload
home_folder = r"C:\Users\kjhm285\Downloads\test_jiayi_2_download-166618297675.tar\test_jiayi_2_download-166618297675"

defect_map_path = os.path.join(home_folder, 'defect_map.json')

train_image_folder = os.path.join(home_folder, 'train', 'Images')
train_segmentation_folder = os.path.join(home_folder, 'train', 'Segmentations')

image_file_name = "2022-10-19T02-43-45-229Z-00f75a62-edaa-4e4e-a4fa-23a85da8fffa.bmp"
mask_file_name = "2022-10-19T02-43-45-229Z-00f75a62-edaa-4e4e-a4fa-23a85da8fffa.png"

train_image_path = os.path.join(train_image_folder, image_file_name)
train_mask_path = os.path.join(train_segmentation_folder, mask_file_name)

res=llens.media.upload(path = train_image_path, split = 'train',
                       seg_mask=train_mask_path, seg_defect_map= defect_map_path,
                       validate_extensions=True, project_id = '12203')

# Summarize after uploading
llens = LandingLens()
media_list = llens.media.ls()
print("After:", media_list)


# if you upload an image that is already in the project, the following error will occur
# Traceback (most recent call last):
#   File "<input>", line 22, in <module>
#   File "C:\ProgramData\Anaconda3\envs\llens\lib\site-packages\landinglens\data_management\media.py", line 398, in upload
#     resp = loop.run_until_complete(tasks)
#   File "C:\ProgramData\Anaconda3\envs\llens\lib\asyncio\base_events.py", line 488, in run_until_complete
#     return future.result()
#   File "C:\ProgramData\Anaconda3\envs\llens\lib\site-packages\landinglens\data_management\media.py", line 244, in upload_media
#     resp_with_content={"filename": filename, "filetype": filetype, "uploadType": "dataset", "projectId": project_id, "md5": media_md5, "uploadId": dataset_id},
#   File "C:\ProgramData\Anaconda3\envs\llens\lib\site-packages\landinglens\client.py", line 126, in _api_async
#     "HTTP request to LandingLens server failed with "
# landinglens.errors.LLensRequestError: HTTP request to LandingLens server failed with code 409-Conflict and error message:
# This image has already been uploaded.






# update the information for one image
from landinglens import LandingLens

# Instantiate the llens client
llens = LandingLens()

# Suppose the image name is 'image2.png', and you need to change its defect type to 5
file_name = '2022-10-19T02-43-45-229Z-00f75a62-edaa-4e4e-a4fa-23a85da8fffa.bmp'
defect_type = 5

# Record the updated ones
updated_media_ids = []

# Iterate the media block by block
for media in llens.media.ls(no_pagination=True)['medias']:
    if media['name'] == file_name:
        # Change the defect type
        llens.metadata.upload(media['id'], defect_type=defect_type)
        updated_media_ids.append(media['id'])

# Show the number of the updated media
print("#{} of images have been changed".format(len(updated_media_ids)))






