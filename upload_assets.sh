#!/bin/bash

# 上传Release附件

TOKEN="[REDACTED]"
REPO="Yongming-Duan/pyramex"
RELEASE_ID="286540608"
DIST_DIR="/home/yongming/openclaw/pyramex/dist"

echo "上传附件到GitHub Release..."

# 上传wheel文件
echo "1/2 上传wheel包..."
curl -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Content-Type: application/octet-stream" \
  https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=pyramex-0.1.0-py3-none-any.whl \
  --data-binary @$DIST_DIR/pyramex-0.1.0-py3-none-any.whl

echo ""
echo "wheel包上传完成"

# 上传源码包
echo "2/2 上传源码包..."
curl -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Content-Type: application/gzip" \
  https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=pyramex-0.1.0.tar.gz \
  --data-binary @$DIST_DIR/pyramex-0.1.0.tar.gz

echo ""
echo "源码包上传完成"

echo ""
echo "✅ 所有附件上传完成！"
echo "访问: https://github.com/$REPO/releases/tag/v0.1.0-beta"
