#!/usr/bin/env bash
# 데스크탑 WSL에서 명령을 "분리 실행"한다 (ssh가 끊겨도 계속 돈다).
#
# 왜 필요한가: on-desktop.sh는 `ssh host "wsl -e bash -lc ..."` 형태라,
# ssh가 끊기면 wsl.exe가 종료되고 WSL2 배포판 자체가 내려간다. 그러면 안에서
# 돌던 프로세스는 물론 tmux 서버까지 통째로 사라진다. nohup도 tmux도 소용없다.
#
# 해결: Windows 쪽에서 Start-Process로 wsl.exe를 **분리된 프로세스**로 띄운다.
# 그 wsl.exe가 살아 있는 동안 WSL 배포판이 유지되므로 안의 작업도 계속 돈다.
#
# 사용: on-desktop-detached.sh "cd ~/work && ./long_job.sh > out.log 2>&1"
set -euo pipefail
DESKTOP_HOST="hwdesktop"

if [ $# -eq 0 ]; then echo "usage: $0 <command>" >&2; exit 2; fi

# ssh -> cmd.exe -> powershell -> wsl -> bash 5단 중첩 인용을 피하려고
# 원본 명령을 base64로 감싼다 (영숫자+/=뿐이라 어느 셸도 건드리지 않는다).
encoded=$(printf '%s' "$*" | base64 -w0)

ssh -o BatchMode=yes "$DESKTOP_HOST" \
  "powershell -NoProfile -Command \"Start-Process -WindowStyle Hidden -FilePath wsl -ArgumentList '-e','bash','-lc','echo $encoded | base64 -d | bash -l'\""
