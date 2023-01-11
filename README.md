# Seminar

Intelligence Engineering Lab 세미나

---

발표자 가이드


### Preliminaries
1. [git](https://git-scm.com/) 설치 

2. github 가입 후 rlaalstjr47 AT korea.ac.kr 로 계정정보 전송 

3. 관리자의 collaborator로 승격 대기

### Git 시작하기

collaborator 자격을 얻었다면 git clone으로 github의 원격 repository를 복사해보자.

```bash
(base) ielab@user2:~/seminar$ git clone https://github.com/Intelligence-Engineering-LAB-KU/Seminar
'Seminar'에 복제합니다...
warning: 빈 저장소를 복제한 것처럼 보입니다.
(base) ielab@user2:~/seminar$ ls
Seminar
(base) ielab@user2:~/seminar$ 
```

### 파일 Post 하기

1. 추가할 파일을 원하는 위치에 옮겨 넣기

```bash
(base) ielab@user2:~/seminar$ cd Seminar/
(base) ielab@user2:~/seminar/Seminar$ mkdir summer_2020
(base) ielab@user2:~/seminar/Seminar$ mv ~/Fourier\ analysis.ipynb summer_2020/
(base) ielab@user2:~/seminar/Seminar$ ls
summer_2020
(base) ielab@user2:~/seminar/Seminar$ ls summer_2020/
'Fourier analysis.ipynb'
(base) ielab@user2:~/seminar/Seminar$ 
```

2. commit 하기

```bash
(base) ielab@user2:~/seminar/Seminar$ git add -A
(base) ielab@user2:~/seminar/Seminar$ git commit -m 'post a file'
[master (최상위-커밋) 494aee2] post a file
 1 file changed, 30139 insertions(+)
 create mode 100644 summer_2020/Fourier analysis.ipynb
```

3. github으로 push하기

```bash
(base) ielab@user2:~/seminar/Seminar$ git push origin master
Username for 'https://github.com': ws_choi@korea.ac.kr
Password for 'https://ws_choi@korea.ac.kr@github.com': 
오브젝트 개수 세는 중: 4, 완료.
Delta compression using up to 20 threads.
오브젝트 압축하는 중: 100% (3/3), 완료.
오브젝트 쓰는 중: 100% (4/4), 1.77 MiB | 1.42 MiB/s, 완료.
Total 4 (delta 0), reused 0 (delta 0)
To https://github.com/Intelligence-Engineering-LAB-KU/Seminar
 * [new branch]      master -> master
(base) ielab@user2:~/seminar/Seminar$ 
```
