// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkKeyContainer_H
#define _C_CkKeyContainer_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkKeyContainer CkKeyContainer_Create(void);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_Dispose(HCkKeyContainer handle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_getContainerName(HCkKeyContainer cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_containerName(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_getDebugLogFilePath(HCkKeyContainer cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_putDebugLogFilePath(HCkKeyContainer cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_debugLogFilePath(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_getIsMachineKeyset(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_getIsOpen(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_getLastErrorHtml(HCkKeyContainer cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_lastErrorHtml(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_getLastErrorText(HCkKeyContainer cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_lastErrorText(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_getLastErrorXml(HCkKeyContainer cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_lastErrorXml(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_getLastMethodSuccess(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_putLastMethodSuccess(HCkKeyContainer cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkKeyContainer_getLegacyKeySpec(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_putLegacyKeySpec(HCkKeyContainer cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_getMachineKeys(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_putMachineKeys(HCkKeyContainer cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_getUtf8(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_putUtf8(HCkKeyContainer cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_getVerboseLogging(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_putVerboseLogging(HCkKeyContainer cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_getVersion(HCkKeyContainer cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_version(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC void CkKeyContainer_CloseContainer(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_CreateContainer(HCkKeyContainer cHandle, const char *name, BOOL machineKeyset);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_DeleteContainer(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_DeleteKey(HCkKeyContainer cHandle, const char *keyName, const char *storageProvider);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_ExportKey(HCkKeyContainer cHandle, const char *keyName, const char *storageProvider, BOOL silentFlag, HCkPrivateKey key);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_FetchContainerNames(HCkKeyContainer cHandle, BOOL bMachineKeyset);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_GenerateKeyPair(HCkKeyContainer cHandle, BOOL bKeyExchangePair, int keyLengthInBits);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_GenerateUuid(HCkKeyContainer cHandle, HCkString outGuid);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_generateUuid(HCkKeyContainer cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_GetKeys(HCkKeyContainer cHandle, const char *storageProvider, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_GetNthContainerName(HCkKeyContainer cHandle, BOOL bMachineKeyset, int index, HCkString outName);
CK_C_VISIBLE_PUBLIC const char *CkKeyContainer_getNthContainerName(HCkKeyContainer cHandle, BOOL bMachineKeyset, int index);
CK_C_VISIBLE_PUBLIC int CkKeyContainer_GetNumContainers(HCkKeyContainer cHandle, BOOL bMachineKeyset);
CK_C_VISIBLE_PUBLIC HCkPrivateKey CkKeyContainer_GetPrivateKey(HCkKeyContainer cHandle, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC HCkPublicKey CkKeyContainer_GetPublicKey(HCkKeyContainer cHandle, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_GetStorageProviders(HCkKeyContainer cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_ImportKey(HCkKeyContainer cHandle, const char *keyName, const char *storageProvider, BOOL allowOverwrite, BOOL allowExport, HCkPrivateKey key);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_ImportPrivateKey(HCkKeyContainer cHandle, HCkPrivateKey key, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_ImportPublicKey(HCkKeyContainer cHandle, HCkPublicKey key, BOOL bKeyExchangePair);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_OpenContainer(HCkKeyContainer cHandle, const char *name, BOOL needPrivateKeyAccess, BOOL machineKeyset);
CK_C_VISIBLE_PUBLIC BOOL CkKeyContainer_SaveLastError(HCkKeyContainer cHandle, const char *path);
#endif
