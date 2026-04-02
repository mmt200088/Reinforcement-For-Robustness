# 日志说明（Logs）

`output.log` 体积约 200MB，超过 GitHub 单文件约 100MB 限制，因此仓库中仅保留 **`output.log.gz`**（内容与 `output.log` 相同，经 gzip 压缩）。

在本地还原完整文本日志：

```bash
gzip -dc output.log.gz > output.log
```

English: `output.log` is not stored in Git due to GitHub’s file size limit; use `output.log.gz` and decompress as above to recover the original log.
