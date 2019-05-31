# Spine-analyzer-project

git - 간편 안내서
https://rogerdudler.github.io/git-guide/index.ko.html

1. coco_mpi_apis.py

COCO와 MPII를 모두 사용하고 어떤 것을 채택할지 선택하기 위해서 작성
API로 COCO와 MPII를, MODE로 동영상과 웹캠을 선택해서 실행시킬 수 있다.
결과물로 같은 파일에 output.avi라는 자세인식을 하고 특정부위에 점을 찍은
동영상을 저장한다.  

2. coco_saveJson.py

API로 COCO만을 사용한다. 
결과물로 output.avi와 YYYY-MM-DD.json, 첫 이미지와 마지막 이미지를 
현재 파일에 저장한다.
