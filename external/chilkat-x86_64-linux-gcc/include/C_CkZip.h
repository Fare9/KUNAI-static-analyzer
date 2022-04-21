// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkZip_H
#define _C_CkZip_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkZip_setAbortCheck(HCkZip cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkZip_setPercentDone(HCkZip cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkZip_setProgressInfo(HCkZip cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkZip_setTaskCompleted(HCkZip cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkZip_setAbortCheck2(HCkZip cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkZip_setPercentDone2(HCkZip cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkZip_setProgressInfo2(HCkZip cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkZip_setTaskCompleted2(HCkZip cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkZip_setExternalProgress(HCkZip cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkZip_setCallbackContext(HCkZip cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkZip CkZip_Create(void);
CK_C_VISIBLE_PUBLIC void CkZip_Dispose(HCkZip handle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getAbortCurrent(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putAbortCurrent(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getAppendFromDir(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putAppendFromDir(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_appendFromDir(HCkZip cHandle);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getAutoRun(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putAutoRun(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_autoRun(HCkZip cHandle);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getAutoRunParams(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putAutoRunParams(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_autoRunParams(HCkZip cHandle);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_getAutoTemp(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putAutoTemp(HCkZip cHandle, BOOL newVal);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkZip_getCaseSensitive(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putCaseSensitive(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getClearArchiveAttribute(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putClearArchiveAttribute(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getClearReadOnlyAttr(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putClearReadOnlyAttr(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getComment(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putComment(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_comment(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_getDebugLogFilePath(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putDebugLogFilePath(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_debugLogFilePath(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_getDecryptPassword(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putDecryptPassword(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_decryptPassword(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getDiscardPaths(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putDiscardPaths(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkZip_getEncryption(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putEncryption(HCkZip cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkZip_getEncryptKeyLength(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putEncryptKeyLength(HCkZip cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getEncryptPassword(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putEncryptPassword(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_encryptPassword(HCkZip cHandle);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getExeDefaultDir(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putExeDefaultDir(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_exeDefaultDir(HCkZip cHandle);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_getExeFinishNotifier(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putExeFinishNotifier(HCkZip cHandle, BOOL newVal);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getExeIconFile(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putExeIconFile(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_exeIconFile(HCkZip cHandle);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_getExeNoInterface(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putExeNoInterface(HCkZip cHandle, BOOL newVal);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_getExeSilentProgress(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putExeSilentProgress(HCkZip cHandle, BOOL newVal);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getExeTitle(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putExeTitle(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_exeTitle(HCkZip cHandle);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getExeUnzipCaption(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putExeUnzipCaption(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_exeUnzipCaption(HCkZip cHandle);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getExeUnzipDir(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putExeUnzipDir(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_exeUnzipDir(HCkZip cHandle);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_getExeWaitForSetup(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putExeWaitForSetup(HCkZip cHandle, BOOL newVal);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_getExeXmlConfig(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putExeXmlConfig(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_exeXmlConfig(HCkZip cHandle);
#endif
CK_C_VISIBLE_PUBLIC int CkZip_getFileCount(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_getFileName(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putFileName(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_fileName(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getHasZipFormatErrors(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC int CkZip_getHeartbeatMs(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putHeartbeatMs(HCkZip cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getIgnoreAccessDenied(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putIgnoreAccessDenied(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getLastErrorHtml(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZip_lastErrorHtml(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_getLastErrorText(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZip_lastErrorText(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_getLastErrorXml(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZip_lastErrorXml(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getLastMethodSuccess(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putLastMethodSuccess(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkZip_getNumEntries(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC int CkZip_getOemCodePage(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putOemCodePage(HCkZip cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getOverwriteExisting(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putOverwriteExisting(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getPasswordProtect(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putPasswordProtect(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getPathPrefix(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putPathPrefix(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_pathPrefix(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC int CkZip_getPercentDoneScale(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putPercentDoneScale(HCkZip cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getPwdProtCharset(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putPwdProtCharset(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_pwdProtCharset(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_getTempDir(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putTempDir(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_tempDir(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getTextFlag(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putTextFlag(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getUncommonOptions(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putUncommonOptions(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_uncommonOptions(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getUtf8(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putUtf8(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getVerboseLogging(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putVerboseLogging(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getVersion(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZip_version(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_getZipx(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC void CkZip_putZipx(HCkZip cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZip_getZipxDefaultAlg(HCkZip cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZip_putZipxDefaultAlg(HCkZip cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZip_zipxDefaultAlg(HCkZip cHandle);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_AddEmbedded(HCkZip cHandle, const char *exeFilename, const char *resourceName, const char *zipFilename);
#endif
CK_C_VISIBLE_PUBLIC void CkZip_AddNoCompressExtension(HCkZip cHandle, const char *fileExtension);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendBase64(HCkZip cHandle, const char *fileName, const char *encodedCompressedData);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendBd(HCkZip cHandle, const char *pathInZip, HCkBinData byteData);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendCompressed(HCkZip cHandle, const char *filename, HCkByteData inData);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendData(HCkZip cHandle, const char *fileName, HCkByteData inData);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendDataEncoded(HCkZip cHandle, const char *filename, const char *encoding, const char *data);
CK_C_VISIBLE_PUBLIC BOOL CkZip_AppendFiles(HCkZip cHandle, const char *filePattern, BOOL recurse);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_AppendFilesAsync(HCkZip cHandle, const char *filePattern, BOOL recurse);
CK_C_VISIBLE_PUBLIC BOOL CkZip_AppendFilesEx(HCkZip cHandle, const char *filePattern, BOOL recurse, BOOL saveExtraPath, BOOL archiveOnly, BOOL includeHidden, BOOL includeSystem);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_AppendFilesExAsync(HCkZip cHandle, const char *filePattern, BOOL recurse, BOOL saveExtraPath, BOOL archiveOnly, BOOL includeHidden, BOOL includeSystem);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendHex(HCkZip cHandle, const char *fileName, const char *encodedCompressedData);
CK_C_VISIBLE_PUBLIC BOOL CkZip_AppendMultiple(HCkZip cHandle, HCkStringArray fileSpecs, BOOL recurse);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_AppendMultipleAsync(HCkZip cHandle, HCkStringArray fileSpecs, BOOL recurse);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendNew(HCkZip cHandle, const char *fileName);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendNewDir(HCkZip cHandle, const char *dirName);
CK_C_VISIBLE_PUBLIC BOOL CkZip_AppendOneFileOrDir(HCkZip cHandle, const char *fileOrDirPath, BOOL saveExtraPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_AppendOneFileOrDirAsync(HCkZip cHandle, const char *fileOrDirPath, BOOL saveExtraPath);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendString(HCkZip cHandle, const char *internalZipFilepath, const char *textData);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_AppendString2(HCkZip cHandle, const char *internalZipFilepath, const char *textData, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkZip_AppendZip(HCkZip cHandle, const char *zipFileName);
CK_C_VISIBLE_PUBLIC void CkZip_CloseZip(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_DeleteEntry(HCkZip cHandle, HCkZipEntry entry);
CK_C_VISIBLE_PUBLIC void CkZip_ExcludeDir(HCkZip cHandle, const char *dirName);
CK_C_VISIBLE_PUBLIC BOOL CkZip_Extract(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_ExtractAsync(HCkZip cHandle, const char *dirPath);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_ExtractExe(HCkZip cHandle, const char *exePath, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_ExtractExeAsync(HCkZip cHandle, const char *exePath, const char *dirPath);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkZip_ExtractInto(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC BOOL CkZip_ExtractMatching(HCkZip cHandle, const char *dirPath, const char *pattern);
CK_C_VISIBLE_PUBLIC BOOL CkZip_ExtractNewer(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC BOOL CkZip_ExtractOne(HCkZip cHandle, HCkZipEntry entry, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_FirstEntry(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_FirstMatchingEntry(HCkZip cHandle, const char *pattern);
CK_C_VISIBLE_PUBLIC BOOL CkZip_GetDirectoryAsXML(HCkZip cHandle, HCkString outXml);
CK_C_VISIBLE_PUBLIC const char *CkZip_getDirectoryAsXML(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_GetEntryByID(HCkZip cHandle, int entryID);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_GetEntryByIndex(HCkZip cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_GetEntryByName(HCkZip cHandle, const char *entryName);
CK_C_VISIBLE_PUBLIC HCkStringArray CkZip_GetExclusions(HCkZip cHandle);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_GetExeConfigParam(HCkZip cHandle, const char *name, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkZip_getExeConfigParam(HCkZip cHandle, const char *name);
#endif
CK_C_VISIBLE_PUBLIC HCkZipEntry CkZip_InsertNew(HCkZip cHandle, const char *fileName, int beforeIndex);
CK_C_VISIBLE_PUBLIC BOOL CkZip_IsNoCompressExtension(HCkZip cHandle, const char *fileExtension);
CK_C_VISIBLE_PUBLIC BOOL CkZip_IsPasswordProtected(HCkZip cHandle, const char *zipFilename);
CK_C_VISIBLE_PUBLIC BOOL CkZip_IsUnlocked(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_LoadTaskCaller(HCkZip cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkZip_NewZip(HCkZip cHandle, const char *zipFilePath);
CK_C_VISIBLE_PUBLIC BOOL CkZip_OpenBd(HCkZip cHandle, HCkBinData binData);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_OpenEmbedded(HCkZip cHandle, const char *exeFilename, const char *resourceName);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkZip_OpenFromByteData(HCkZip cHandle, HCkByteData byteData);
CK_C_VISIBLE_PUBLIC BOOL CkZip_OpenFromMemory(HCkZip cHandle, HCkByteData inData);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_OpenMyEmbedded(HCkZip cHandle, const char *resourceName);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkZip_OpenZip(HCkZip cHandle, const char *zipPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_OpenZipAsync(HCkZip cHandle, const char *zipPath);
CK_C_VISIBLE_PUBLIC BOOL CkZip_QuickAppend(HCkZip cHandle, const char *ZipFileName);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_QuickAppendAsync(HCkZip cHandle, const char *ZipFileName);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_RemoveEmbedded(HCkZip cHandle, const char *exeFilename, const char *resourceName);
#endif
CK_C_VISIBLE_PUBLIC void CkZip_RemoveNoCompressExtension(HCkZip cHandle, const char *fileExtension);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_ReplaceEmbedded(HCkZip cHandle, const char *exeFilename, const char *resourceName, const char *zipFilename);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkZip_SaveLastError(HCkZip cHandle, const char *path);
CK_C_VISIBLE_PUBLIC void CkZip_SetCompressionLevel(HCkZip cHandle, int level);
CK_C_VISIBLE_PUBLIC void CkZip_SetExclusions(HCkZip cHandle, HCkStringArray excludePatterns);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC void CkZip_SetExeConfigParam(HCkZip cHandle, const char *paramName, const char *paramValue);
#endif
CK_C_VISIBLE_PUBLIC void CkZip_SetPassword(HCkZip cHandle, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkZip_UnlockComponent(HCkZip cHandle, const char *regCode);
CK_C_VISIBLE_PUBLIC int CkZip_Unzip(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_UnzipAsync(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC int CkZip_UnzipInto(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_UnzipIntoAsync(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC int CkZip_UnzipMatching(HCkZip cHandle, const char *dirPath, const char *pattern, BOOL verbose);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_UnzipMatchingAsync(HCkZip cHandle, const char *dirPath, const char *pattern, BOOL verbose);
CK_C_VISIBLE_PUBLIC int CkZip_UnzipMatchingInto(HCkZip cHandle, const char *dirPath, const char *pattern, BOOL verbose);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_UnzipMatchingIntoAsync(HCkZip cHandle, const char *dirPath, const char *pattern, BOOL verbose);
CK_C_VISIBLE_PUBLIC int CkZip_UnzipNewer(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_UnzipNewerAsync(HCkZip cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC BOOL CkZip_VerifyPassword(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_WriteBd(HCkZip cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_WriteBdAsync(HCkZip cHandle, HCkBinData binData);
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_WriteExe(HCkZip cHandle, const char *exeFilename);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_WriteExe2(HCkZip cHandle, const char *exePath, const char *destExePath, BOOL bAesEncrypt, int keyLength, const char *password);
#endif
#if defined(CK_SFX_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkZip_WriteExeToMemory(HCkZip cHandle, HCkByteData outBytes);
#endif
CK_C_VISIBLE_PUBLIC BOOL CkZip_WriteToMemory(HCkZip cHandle, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_WriteToMemoryAsync(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_WriteZip(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_WriteZipAsync(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZip_WriteZipAndClose(HCkZip cHandle);
CK_C_VISIBLE_PUBLIC HCkTask CkZip_WriteZipAndCloseAsync(HCkZip cHandle);
#endif
