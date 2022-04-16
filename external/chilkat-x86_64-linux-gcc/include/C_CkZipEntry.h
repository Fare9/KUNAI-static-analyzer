// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkZipEntry_H
#define _C_CkZipEntry_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkZipEntry_setAbortCheck(HCkZipEntry cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkZipEntry_setPercentDone(HCkZipEntry cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkZipEntry_setProgressInfo(HCkZipEntry cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkZipEntry_setTaskCompleted(HCkZipEntry cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkZipEntry_setAbortCheck2(HCkZipEntry cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkZipEntry_setPercentDone2(HCkZipEntry cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkZipEntry_setProgressInfo2(HCkZipEntry cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkZipEntry_setTaskCompleted2(HCkZipEntry cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkZipEntry_setExternalProgress(HCkZipEntry cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkZipEntry_setCallbackContext(HCkZipEntry cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkZipEntry CkZipEntry_Create(void);
CK_C_VISIBLE_PUBLIC void CkZipEntry_Dispose(HCkZipEntry handle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getComment(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putComment(HCkZipEntry cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_comment(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkZipEntry_getCompressedLength(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC __int64 CkZipEntry_getCompressedLength64(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getCompressedLengthStr(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_compressedLengthStr(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntry_getCompressionLevel(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putCompressionLevel(HCkZipEntry cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkZipEntry_getCompressionMethod(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putCompressionMethod(HCkZipEntry cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkZipEntry_getCrc(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getDebugLogFilePath(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putDebugLogFilePath(HCkZipEntry cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_debugLogFilePath(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntry_getEncryptionKeyLen(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntry_getEntryID(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntry_getEntryType(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getFileDateTime(HCkZipEntry cHandle, SYSTEMTIME * retval);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putFileDateTime(HCkZipEntry cHandle, SYSTEMTIME *newVal);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getFileDateTimeStr(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putFileDateTimeStr(HCkZipEntry cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_fileDateTimeStr(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getFileName(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putFileName(HCkZipEntry cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_fileName(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getFileNameHex(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_fileNameHex(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntry_getHeartbeatMs(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putHeartbeatMs(HCkZipEntry cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_getIsAesEncrypted(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_getIsDirectory(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getLastErrorHtml(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_lastErrorHtml(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getLastErrorText(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_lastErrorText(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getLastErrorXml(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_lastErrorXml(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_getLastMethodSuccess(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putLastMethodSuccess(HCkZipEntry cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_getTextFlag(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putTextFlag(HCkZipEntry cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC unsigned long CkZipEntry_getUncompressedLength(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC __int64 CkZipEntry_getUncompressedLength64(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getUncompressedLengthStr(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_uncompressedLengthStr(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_getUtf8(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putUtf8(HCkZipEntry cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_getVerboseLogging(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntry_putVerboseLogging(HCkZipEntry cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZipEntry_getVersion(HCkZipEntry cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_version(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_AppendData(HCkZipEntry cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_AppendDataAsync(HCkZipEntry cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_AppendString(HCkZipEntry cHandle, const char *strContent, const char *charset);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_AppendStringAsync(HCkZipEntry cHandle, const char *strContent, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_Copy(HCkZipEntry cHandle, HCkByteData outData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_CopyToBase64(HCkZipEntry cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_copyToBase64(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_CopyToHex(HCkZipEntry cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_copyToHex(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_Extract(HCkZipEntry cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_ExtractAsync(HCkZipEntry cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_ExtractInto(HCkZipEntry cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_ExtractIntoAsync(HCkZipEntry cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkDateTime CkZipEntry_GetDt(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_Inflate(HCkZipEntry cHandle, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_InflateAsync(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_LoadTaskCaller(HCkZipEntry cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZipEntry_NextEntry(HCkZipEntry cHandle);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZipEntry_NextMatchingEntry(HCkZipEntry cHandle, const char *matchStr);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_ReplaceData(HCkZipEntry cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_ReplaceString(HCkZipEntry cHandle, const char *strContent, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_SaveLastError(HCkZipEntry cHandle, const char *path);
CK_C_VISIBLE_PUBLIC void CkZipEntry_SetDt(HCkZipEntry cHandle, HCkDateTime dt);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_UnzipToBd(HCkZipEntry cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_UnzipToBdAsync(HCkZipEntry cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_UnzipToSb(HCkZipEntry cHandle, int lineEndingBehavior, const char *srcCharset, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_UnzipToSbAsync(HCkZipEntry cHandle, int lineEndingBehavior, const char *srcCharset, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_UnzipToStream(HCkZipEntry cHandle, HCkStream toStream);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_UnzipToStreamAsync(HCkZipEntry cHandle, HCkStream toStream);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntry_UnzipToString(HCkZipEntry cHandle, int lineEndingBehavior, const char *srcCharset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkZipEntry_unzipToString(HCkZipEntry cHandle, int lineEndingBehavior, const char *srcCharset);
CK_C_VISIBLE_PUBLIC HCkTask CkZipEntry_UnzipToStringAsync(HCkZipEntry cHandle, int lineEndingBehavior, const char *srcCharset);
#endif
