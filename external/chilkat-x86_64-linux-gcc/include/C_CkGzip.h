// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkGzip_H
#define _C_CkGzip_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkGzip_setAbortCheck(HCkGzip cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkGzip_setPercentDone(HCkGzip cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkGzip_setProgressInfo(HCkGzip cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkGzip_setTaskCompleted(HCkGzip cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkGzip_setAbortCheck2(HCkGzip cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkGzip_setPercentDone2(HCkGzip cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkGzip_setProgressInfo2(HCkGzip cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkGzip_setTaskCompleted2(HCkGzip cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkGzip_setExternalProgress(HCkGzip cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkGzip_setCallbackContext(HCkGzip cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkGzip CkGzip_Create(void);
CK_C_VISIBLE_PUBLIC void CkGzip_Dispose(HCkGzip handle);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_getAbortCurrent(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_putAbortCurrent(HCkGzip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkGzip_getComment(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkGzip_putComment(HCkGzip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkGzip_comment(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC int CkGzip_getCompressionLevel(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_putCompressionLevel(HCkGzip cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkGzip_getDebugLogFilePath(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkGzip_putDebugLogFilePath(HCkGzip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkGzip_debugLogFilePath(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_getExtraData(HCkGzip cHandle, HCkByteData retval);
CK_C_VISIBLE_PUBLIC void CkGzip_putExtraData(HCkGzip cHandle, HCkByteData newVal);
CK_C_VISIBLE_PUBLIC void CkGzip_getFilename(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkGzip_putFilename(HCkGzip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkGzip_filename(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC int CkGzip_getHeartbeatMs(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_putHeartbeatMs(HCkGzip cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkGzip_getLastErrorHtml(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkGzip_lastErrorHtml(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_getLastErrorText(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkGzip_lastErrorText(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_getLastErrorXml(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkGzip_lastErrorXml(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_getLastMethodSuccess(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_putLastMethodSuccess(HCkGzip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkGzip_getLastMod(HCkGzip cHandle, SYSTEMTIME * retval);
CK_C_VISIBLE_PUBLIC void CkGzip_putLastMod(HCkGzip cHandle, SYSTEMTIME *newVal);
CK_C_VISIBLE_PUBLIC void CkGzip_getLastModStr(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkGzip_putLastModStr(HCkGzip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkGzip_lastModStr(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_getUseCurrentDate(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_putUseCurrentDate(HCkGzip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_getUtf8(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_putUtf8(HCkGzip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_getVerboseLogging(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC void CkGzip_putVerboseLogging(HCkGzip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkGzip_getVersion(HCkGzip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkGzip_version(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressBd(HCkGzip cHandle, HCkBinData binDat);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressBdAsync(HCkGzip cHandle, HCkBinData binDat);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressFile(HCkGzip cHandle, const char *inFilename, const char *destPath);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressFileAsync(HCkGzip cHandle, const char *inFilename, const char *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressFile2(HCkGzip cHandle, const char *inFilename, const char *embeddedFilename, const char *destPath);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressFile2Async(HCkGzip cHandle, const char *inFilename, const char *embeddedFilename, const char *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressFileToMem(HCkGzip cHandle, const char *inFilename, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressFileToMemAsync(HCkGzip cHandle, const char *inFilename);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressMemory(HCkGzip cHandle, HCkByteData inData, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressMemoryAsync(HCkGzip cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressMemToFile(HCkGzip cHandle, HCkByteData inData, const char *destPath);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressMemToFileAsync(HCkGzip cHandle, HCkByteData inData, const char *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressString(HCkGzip cHandle, const char *inStr, const char *destCharset, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressStringAsync(HCkGzip cHandle, const char *inStr, const char *destCharset);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressStringENC(HCkGzip cHandle, const char *inStr, const char *charset, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_compressStringENC(HCkGzip cHandle, const char *inStr, const char *charset, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_CompressStringToFile(HCkGzip cHandle, const char *inStr, const char *destCharset, const char *destPath);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_CompressStringToFileAsync(HCkGzip cHandle, const char *inStr, const char *destCharset, const char *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_Decode(HCkGzip cHandle, const char *encodedStr, const char *encoding, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_DeflateStringENC(HCkGzip cHandle, const char *inString, const char *charsetName, const char *outputEncoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_deflateStringENC(HCkGzip cHandle, const char *inString, const char *charsetName, const char *outputEncoding);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_Encode(HCkGzip cHandle, HCkByteData byteData, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_encode(HCkGzip cHandle, HCkByteData byteData, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_ExamineFile(HCkGzip cHandle, const char *inGzFilename);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_ExamineMemory(HCkGzip cHandle, HCkByteData inGzData);
CK_C_VISIBLE_PUBLIC HCkDateTime CkGzip_GetDt(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_InflateStringENC(HCkGzip cHandle, const char *inString, const char *convertFromCharset, const char *inputEncoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_inflateStringENC(HCkGzip cHandle, const char *inString, const char *convertFromCharset, const char *inputEncoding);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_IsUnlocked(HCkGzip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_LoadTaskCaller(HCkGzip cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_ReadFile(HCkGzip cHandle, const char *path, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_SaveLastError(HCkGzip cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_SetDt(HCkGzip cHandle, HCkDateTime dt);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressBd(HCkGzip cHandle, HCkBinData binDat);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UncompressBdAsync(HCkGzip cHandle, HCkBinData binDat);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressFile(HCkGzip cHandle, const char *srcPath, const char *destPath);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UncompressFileAsync(HCkGzip cHandle, const char *srcPath, const char *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressFileToMem(HCkGzip cHandle, const char *inFilename, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UncompressFileToMemAsync(HCkGzip cHandle, const char *inFilename);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressFileToString(HCkGzip cHandle, const char *gzFilename, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_uncompressFileToString(HCkGzip cHandle, const char *gzFilename, const char *charset);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UncompressFileToStringAsync(HCkGzip cHandle, const char *gzFilename, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressMemory(HCkGzip cHandle, HCkByteData inData, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UncompressMemoryAsync(HCkGzip cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressMemToFile(HCkGzip cHandle, HCkByteData inData, const char *destPath);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UncompressMemToFileAsync(HCkGzip cHandle, HCkByteData inData, const char *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressString(HCkGzip cHandle, HCkByteData inData, const char *inCharset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_uncompressString(HCkGzip cHandle, HCkByteData inData, const char *inCharset);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UncompressStringAsync(HCkGzip cHandle, HCkByteData inData, const char *inCharset);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UncompressStringENC(HCkGzip cHandle, const char *inStr, const char *charset, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_uncompressStringENC(HCkGzip cHandle, const char *inStr, const char *charset, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UnlockComponent(HCkGzip cHandle, const char *unlockCode);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_UnTarGz(HCkGzip cHandle, const char *tgzFilename, const char *destDir, BOOL bNoAbsolute);
CK_C_VISIBLE_PUBLIC HCkTask CkGzip_UnTarGzAsync(HCkGzip cHandle, const char *tgzFilename, const char *destDir, BOOL bNoAbsolute);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_WriteFile(HCkGzip cHandle, const char *path, HCkByteData binaryData);
CK_C_VISIBLE_PUBLIC BOOL CkGzip_XfdlToXml(HCkGzip cHandle, const char *xfldData, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkGzip_xfdlToXml(HCkGzip cHandle, const char *xfldData);
#endif
