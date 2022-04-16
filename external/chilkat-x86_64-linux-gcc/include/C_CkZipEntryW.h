// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkZipEntryWH
#define _C_CkZipEntryWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkZipEntryW_setAbortCheck(HCkZipEntryW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkZipEntryW_setPercentDone(HCkZipEntryW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkZipEntryW_setProgressInfo(HCkZipEntryW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkZipEntryW_setTaskCompleted(HCkZipEntryW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkZipEntryW CkZipEntryW_Create(void);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_Dispose(HCkZipEntryW handle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getComment(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putComment(HCkZipEntryW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_comment(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkZipEntryW_getCompressedLength(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC __int64 CkZipEntryW_getCompressedLength64(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getCompressedLengthStr(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_compressedLengthStr(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntryW_getCompressionLevel(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putCompressionLevel(HCkZipEntryW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkZipEntryW_getCompressionMethod(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putCompressionMethod(HCkZipEntryW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkZipEntryW_getCrc(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getDebugLogFilePath(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putDebugLogFilePath(HCkZipEntryW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_debugLogFilePath(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntryW_getEncryptionKeyLen(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntryW_getEntryID(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntryW_getEntryType(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getFileDateTime(HCkZipEntryW cHandle, SYSTEMTIME * retval);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putFileDateTime(HCkZipEntryW cHandle, SYSTEMTIME *newVal);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getFileDateTimeStr(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putFileDateTimeStr(HCkZipEntryW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_fileDateTimeStr(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getFileName(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putFileName(HCkZipEntryW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_fileName(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getFileNameHex(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_fileNameHex(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC int CkZipEntryW_getHeartbeatMs(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putHeartbeatMs(HCkZipEntryW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_getIsAesEncrypted(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_getIsDirectory(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getLastErrorHtml(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_lastErrorHtml(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getLastErrorText(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_lastErrorText(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getLastErrorXml(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_lastErrorXml(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_getLastMethodSuccess(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putLastMethodSuccess(HCkZipEntryW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_getTextFlag(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putTextFlag(HCkZipEntryW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC unsigned long CkZipEntryW_getUncompressedLength(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC __int64 CkZipEntryW_getUncompressedLength64(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getUncompressedLengthStr(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_uncompressedLengthStr(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_getVerboseLogging(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipEntryW_putVerboseLogging(HCkZipEntryW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_getVersion(HCkZipEntryW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_version(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_AppendData(HCkZipEntryW cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_AppendDataAsync(HCkZipEntryW cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_AppendString(HCkZipEntryW cHandle, const wchar_t *strContent, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_AppendStringAsync(HCkZipEntryW cHandle, const wchar_t *strContent, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_Copy(HCkZipEntryW cHandle, HCkByteData outData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_CopyToBase64(HCkZipEntryW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_copyToBase64(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_CopyToHex(HCkZipEntryW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_copyToHex(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_Extract(HCkZipEntryW cHandle, const wchar_t *dirPath);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_ExtractAsync(HCkZipEntryW cHandle, const wchar_t *dirPath);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_ExtractInto(HCkZipEntryW cHandle, const wchar_t *dirPath);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_ExtractIntoAsync(HCkZipEntryW cHandle, const wchar_t *dirPath);
CK_C_VISIBLE_PUBLIC HCkDateTimeW CkZipEntryW_GetDt(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_Inflate(HCkZipEntryW cHandle, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_InflateAsync(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_LoadTaskCaller(HCkZipEntryW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC HCkZipEntryW CkZipEntryW_NextEntry(HCkZipEntryW cHandle);
CK_C_VISIBLE_PUBLIC HCkZipEntryW CkZipEntryW_NextMatchingEntry(HCkZipEntryW cHandle, const wchar_t *matchStr);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_ReplaceData(HCkZipEntryW cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_ReplaceString(HCkZipEntryW cHandle, const wchar_t *strContent, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_SaveLastError(HCkZipEntryW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC void CkZipEntryW_SetDt(HCkZipEntryW cHandle, HCkDateTimeW dt);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_UnzipToBd(HCkZipEntryW cHandle, HCkBinDataW binData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_UnzipToBdAsync(HCkZipEntryW cHandle, HCkBinDataW binData);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_UnzipToSb(HCkZipEntryW cHandle, int lineEndingBehavior, const wchar_t *srcCharset, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_UnzipToSbAsync(HCkZipEntryW cHandle, int lineEndingBehavior, const wchar_t *srcCharset, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_UnzipToStream(HCkZipEntryW cHandle, HCkStreamW toStream);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_UnzipToStreamAsync(HCkZipEntryW cHandle, HCkStreamW toStream);
CK_C_VISIBLE_PUBLIC BOOL CkZipEntryW_UnzipToString(HCkZipEntryW cHandle, int lineEndingBehavior, const wchar_t *srcCharset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipEntryW_unzipToString(HCkZipEntryW cHandle, int lineEndingBehavior, const wchar_t *srcCharset);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipEntryW_UnzipToStringAsync(HCkZipEntryW cHandle, int lineEndingBehavior, const wchar_t *srcCharset);
#endif
