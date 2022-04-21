// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkScpWH
#define _C_CkScpWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkScpW_setAbortCheck(HCkScpW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkScpW_setPercentDone(HCkScpW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkScpW_setProgressInfo(HCkScpW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkScpW_setTaskCompleted(HCkScpW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkScpW CkScpW_Create(void);
CK_C_VISIBLE_PUBLIC void CkScpW_Dispose(HCkScpW handle);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_getAbortCurrent(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void  CkScpW_putAbortCurrent(HCkScpW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkScpW_getDebugLogFilePath(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putDebugLogFilePath(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_debugLogFilePath(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC int CkScpW_getHeartbeatMs(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void  CkScpW_putHeartbeatMs(HCkScpW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkScpW_getLastErrorHtml(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_lastErrorHtml(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getLastErrorText(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_lastErrorText(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getLastErrorXml(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_lastErrorXml(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_getLastMethodSuccess(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void  CkScpW_putLastMethodSuccess(HCkScpW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkScpW_getPercentDoneScale(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void  CkScpW_putPercentDoneScale(HCkScpW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkScpW_getSendEnv(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putSendEnv(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_sendEnv(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getSyncedFiles(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putSyncedFiles(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_syncedFiles(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getSyncMustMatch(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putSyncMustMatch(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_syncMustMatch(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getSyncMustMatchDir(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putSyncMustMatchDir(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_syncMustMatchDir(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getSyncMustNotMatch(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putSyncMustNotMatch(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_syncMustNotMatch(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getSyncMustNotMatchDir(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putSyncMustNotMatchDir(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_syncMustNotMatchDir(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getUncommonOptions(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putUncommonOptions(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_uncommonOptions(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void CkScpW_getUnixPermOverride(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScpW_putUnixPermOverride(HCkScpW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_unixPermOverride(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_getVerboseLogging(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC void  CkScpW_putVerboseLogging(HCkScpW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkScpW_getVersion(HCkScpW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_version(HCkScpW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_DownloadBd(HCkScpW cHandle, const wchar_t *remotePath, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_DownloadBdAsync(HCkScpW cHandle, const wchar_t *remotePath, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_DownloadBinary(HCkScpW cHandle, const wchar_t *remotePath, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_DownloadBinaryAsync(HCkScpW cHandle, const wchar_t *remotePath);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_DownloadBinaryEncoded(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_downloadBinaryEncoded(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_DownloadBinaryEncodedAsync(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_DownloadFile(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *localPath);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_DownloadFileAsync(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *localPath);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_DownloadString(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScpW_downloadString(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_DownloadStringAsync(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_LoadTaskCaller(HCkScpW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_SaveLastError(HCkScpW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_SyncTreeDownload(HCkScpW cHandle, const wchar_t *remoteRoot, const wchar_t *localRoot, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_SyncTreeDownloadAsync(HCkScpW cHandle, const wchar_t *remoteRoot, const wchar_t *localRoot, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_SyncTreeUpload(HCkScpW cHandle, const wchar_t *localBaseDir, const wchar_t *remoteBaseDir, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_SyncTreeUploadAsync(HCkScpW cHandle, const wchar_t *localBaseDir, const wchar_t *remoteBaseDir, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_UploadBd(HCkScpW cHandle, const wchar_t *remotePath, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_UploadBdAsync(HCkScpW cHandle, const wchar_t *remotePath, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_UploadBinary(HCkScpW cHandle, const wchar_t *remotePath, HCkByteData binData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_UploadBinaryAsync(HCkScpW cHandle, const wchar_t *remotePath, HCkByteData binData);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_UploadBinaryEncoded(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *encodedData, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_UploadBinaryEncodedAsync(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *encodedData, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_UploadFile(HCkScpW cHandle, const wchar_t *localPath, const wchar_t *remotePath);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_UploadFileAsync(HCkScpW cHandle, const wchar_t *localPath, const wchar_t *remotePath);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_UploadString(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *textData, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC HCkTaskW CkScpW_UploadStringAsync(HCkScpW cHandle, const wchar_t *remotePath, const wchar_t *textData, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkScpW_UseSsh(HCkScpW cHandle, HCkSshW sshConnection);
#endif
