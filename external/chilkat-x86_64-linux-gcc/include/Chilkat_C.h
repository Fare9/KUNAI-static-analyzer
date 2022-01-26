
#ifndef _CHILKAT_C
#define _CHILKAT_C

#if !defined(BOOL_IS_TYPEDEF) && !defined(OBJC_BOOL_DEFINED)
#ifndef BOOL
#define BOOL int
#endif
#endif
	
#ifndef TRUE
#define TRUE 1
#endif
	
#ifndef FALSE
#define FALSE 0
#endif	
	
#if !defined(WIN32) && !defined(WINCE)
#include "SystemTime.h"              
#include "FileTime.h"                
#endif                  

#include "ck_inttypes.h"
	
// Use typedefs so we can explicitly see the kind of object pointed
// to by "void *"
	
#ifdef CK_GO_LANG
#define HCkByteData void *
#define HCkString void *
#define HCkCert void *
#define HCkEmail void *
#define HCkEmailBundle void *
#define HCkMailMan void *
#define HCkMailProgress void *
#define HCkPrivateKey void *
#define HCkPublicKey void *
#define HCkCsp void *
#define HCkMime void *
#define HCkKeyContainer void *
#define HCkCertStore void *
#define HCkCreateCS void *
#define HCkBounce void *
#define HCkCharset void *
#define HCkCrypt2 void *
#define HCkCrypt2Progress void *
#define HCkFtp2 void *
#define HCkFtpProgress void *
#define HCkHtmlToXml void *
#define HCkHtmlToText void *
#define HCkHttp void *
#define HCkHttpProgress void *
#define HCkHttpRequest void *
#define HCkHttpResponse void *
#define HCkImap void *
#define HCkImapProgress void *
#define HCkMailboxes void *
#define HCkMessageSet void *
#define HCkMht void *
#define HCkMhtProgress void *
#define HCkRar void *
#define HCkRarEntry void *
#define HCkRsa void *
#define HCkSocket void *
#define HCkSocketProgress void *
#define HCkSpider void *
#define HCkSpiderProgress void *
#define HCkUpload void *
#define HCkCgi void *
#define HCkSettings void *
#define HCkStringArray void *
#define HCkXml void *
#define HCkAtom void *
#define HCkAtomProgress void *
#define HCkRss void *
#define HCkRssProgress void *
#define HCkZip void *
#define HCkZipProgress void *
#define HCkZipEntry void *
#define HCkZipCrc void *
#define HCkCompression void *
#define HCkGzip void *
#define HCkUnixCompress void *
#define HCkSsh void *
#define HCkSshProgress void *
#define HCkSFtp void *
#define HCkSFtpProgress void *
#define HCkSFtpDir void *
#define HCkSFtpFile void *
#define HCkSshKey void *
#define HCkTar void *
#define HCkTarProgress void *
#define HCkBz2 void *
#define HCkBz2Progress void *
#define HCkDh void *
#define HCkDhProgress void *
#define HCkDsa void *
#define HCkDsaProgress void *
#define HCkXmp void *
#define HCkCache void *
#define HCkDkim void *
#define HCkDkimProgress void *
#define HCkFileAccess void *
#define HCkDateTime void *
#define HCkCsv void *
#define HCkSshTunnel void *
#define HCkOmaDrm void *
#define HCkNtlm void *
#define HCkDirTree void *
#define HCkDtObj void *
#define HCkTrustedRoots void *
#define HCkCertChain void *
#define HCkPfx void *
#define HCkXmlCertVault void *
#define HCkLog void *
#define HCkJavaKeyStore void *
#define HCkAsn void *
#define HCkPem void *
#define HCkUrl void *
#define HCkGlobal void *
#define HCkScp void *
#define HCkHashtable void *
#define HCkTask void *
#define HCkTaskChain void *
#define HCkPrng void *
#define HCkEcc void *
#define HCkOAuth1 void *
#define HCkJsonObject void *
#define HCkJsonArray void *
#define HCkStream void *
#define HCkAuthAws void *
#define HCkAuthGoogle void *
#define HCkAuthAzureStorage void *
#define HCkAuthAzureAD void *
#define HCkAuthAzureSAS void *
#define HCkRest void *
#define HCkStringBuilder void *
#define HCkJwt void *
#define HCkServerSentEvent void *
#define HCkOAuth2 void *
#define HCkBinData void *
#define HCkStringTable void *
#define HCkCsr void *
#define HCkJwe void *
#define HCkJws void *
#define HCkAuthUtil void *
#define HCkXmlDSig void *
#define HCkXmlDSigGen void *
#define HCkWebSocket void *
#define HCkSecureString void *
#define HCkPdf void *
#define HCkEdDSA void *
#define HCkPkcs11 void *
#define HCkScMinidriver void *
#define HCkSCard void *

#else
typedef void *HCkByteData;
typedef void *HCkString;
typedef void *HCkCert;
typedef void *HCkEmail;
typedef void *HCkEmailBundle;
typedef void *HCkMailMan;
typedef void *HCkMailProgress;
typedef void *HCkPrivateKey;
typedef void *HCkPublicKey;
typedef void *HCkCsp;
typedef void *HCkMime;
typedef void *HCkKeyContainer;
typedef void *HCkCertStore;
typedef void *HCkCreateCS;
typedef void *HCkBounce;
typedef void *HCkCharset;
typedef void *HCkCrypt2;
typedef void *HCkCrypt2Progress;
typedef void *HCkFtp2;
typedef void *HCkFtpProgress;
typedef void *HCkHtmlToXml;
typedef void *HCkHtmlToText;
typedef void *HCkHttp;
typedef void *HCkHttpProgress;
typedef void *HCkHttpRequest;
typedef void *HCkHttpResponse;
typedef void *HCkImap;
typedef void *HCkImapProgress;
typedef void *HCkMailboxes;
typedef void *HCkMessageSet;
typedef void *HCkMht;
typedef void *HCkMhtProgress;
typedef void *HCkRar;
typedef void *HCkRarEntry;
typedef void *HCkRsa;
typedef void *HCkSocket;
typedef void *HCkSocketProgress;
typedef void *HCkSpider;
typedef void *HCkSpiderProgress;
typedef void *HCkUpload;
typedef void *HCkCgi;
typedef void *HCkSettings;
typedef void *HCkStringArray;
typedef void *HCkXml;
typedef void *HCkAtom;
typedef void *HCkAtomProgress;
typedef void *HCkRss;
typedef void *HCkRssProgress;
typedef void *HCkZip;
typedef void *HCkZipProgress;
typedef void *HCkZipEntry;
typedef void *HCkZipCrc;
typedef void *HCkCompression;
typedef void *HCkGzip;
typedef void *HCkUnixCompress;
typedef void *HCkSsh;
typedef void *HCkSshProgress;
typedef void *HCkSFtp;
typedef void *HCkSFtpProgress;
typedef void *HCkSFtpDir;
typedef void *HCkSFtpFile;
typedef void *HCkSshKey;
typedef void *HCkTar;
typedef void *HCkTarProgress;
typedef void *HCkBz2;
typedef void *HCkBz2Progress;
typedef void *HCkDh;
typedef void *HCkDhProgress;
typedef void *HCkDsa;
typedef void *HCkDsaProgress;
typedef void *HCkXmp;
typedef void *HCkCache;
typedef void *HCkDkim;
typedef void *HCkDkimProgress;
typedef void *HCkFileAccess;
typedef void *HCkDateTime;
typedef void *HCkCsv;
typedef void *HCkSshTunnel;
typedef void *HCkOmaDrm;
typedef void *HCkNtlm;
typedef void *HCkDirTree;
typedef void *HCkDtObj;
typedef void *HCkTrustedRoots;
typedef void *HCkCertChain;
typedef void *HCkPfx;
typedef void *HCkXmlCertVault;
typedef void *HCkLog;
typedef void *HCkJavaKeyStore;
typedef void *HCkAsn;
typedef void *HCkPem;
typedef void *HCkUrl;
typedef void *HCkGlobal;
typedef void *HCkScp;
typedef void *HCkHashtable;
typedef void *HCkTask;
typedef void *HCkTaskChain;
typedef void *HCkPrng;
typedef void *HCkEcc;
typedef void *HCkOAuth1;
typedef void *HCkJsonObject;
typedef void *HCkJsonArray;
typedef void *HCkStream;
typedef void *HCkAuthAws;
typedef void *HCkAuthGoogle;
typedef void *HCkAuthAzureStorage;
typedef void *HCkAuthAzureAD;
typedef void *HCkAuthAzureSAS;
typedef void *HCkRest;
typedef void *HCkStringBuilder;
typedef void *HCkJwt;
typedef void *HCkServerSentEvent;
typedef void *HCkOAuth2;
typedef void *HCkBinData;
typedef void *HCkStringTable;
typedef void *HCkCsr;
typedef void *HCkJwe;
typedef void *HCkJws;
typedef void *HCkAuthUtil;
typedef void *HCkXmlDSig;
typedef void *HCkXmlDSigGen;
typedef void *HCkWebSocket;
typedef void *HCkSecureString;
typedef void *HCkPdf;
typedef void *HCkEdDSA;
typedef void *HCkPkcs11;
typedef void *HCkScMinidriver;
typedef void *HCkSCard;

typedef void *HCkByteDataW;
typedef void *HCkStringW;
typedef void *HCkCertW;
typedef void *HCkEmailW;
typedef void *HCkEmailBundleW;
typedef void *HCkMailManW;
typedef void *HCkMailProgressW;
typedef void *HCkPrivateKeyW;
typedef void *HCkPublicKeyW;
typedef void *HCkCspW;
typedef void *HCkMimeW;
typedef void *HCkKeyContainerW;
typedef void *HCkCertStoreW;
typedef void *HCkCreateCSW;
typedef void *HCkBounceW;
typedef void *HCkCharsetW;
typedef void *HCkCrypt2W;
typedef void *HCkCrypt2ProgressW;
typedef void *HCkFtp2W;
typedef void *HCkFtpProgressW;
typedef void *HCkHtmlToXmlW;
typedef void *HCkHtmlToTextW;
typedef void *HCkHttpW;
typedef void *HCkHttpProgressW;
typedef void *HCkHttpRequestW;
typedef void *HCkHttpResponseW;
typedef void *HCkImapW;
typedef void *HCkImapProgressW;
typedef void *HCkMailboxesW;
typedef void *HCkMessageSetW;
typedef void *HCkMhtW;
typedef void *HCkMhtProgressW;
typedef void *HCkRarW;
typedef void *HCkRarEntryW;
typedef void *HCkRsaW;
typedef void *HCkSocketW;
typedef void *HCkSocketProgressW;
typedef void *HCkSpiderW;
typedef void *HCkSpiderProgressW;
typedef void *HCkUploadW;
typedef void *HCkCgiW;
typedef void *HCkSettingsW;
typedef void *HCkStringArrayW;
typedef void *HCkXmlW;
typedef void *HCkAtomW;
typedef void *HCkAtomProgressW;
typedef void *HCkRssW;
typedef void *HCkRssProgressW;
typedef void *HCkZipW;
typedef void *HCkZipProgressW;
typedef void *HCkZipEntryW;
typedef void *HCkZipCrcW;
typedef void *HCkCompressionW;
typedef void *HCkGzipW;
typedef void *HCkUnixCompressW;
typedef void *HCkSshW;
typedef void *HCkSFtpW;
typedef void *HCkSshProgressW;
typedef void *HCkSFtpProgressW;
typedef void *HCkSFtpDirW;
typedef void *HCkSFtpFileW;
typedef void *HCkSshKeyW;
typedef void *HCkTarW;
typedef void *HCkBz2W;
typedef void *HCkDhW;
typedef void *HCkDsaW;
typedef void *HCkTarProgressW;
typedef void *HCkBz2ProgressW;
typedef void *HCkDhProgressW;
typedef void *HCkDsaProgressW;
typedef void *HCkXmpW;
typedef void *HCkCacheW;
typedef void *HCkDkimW;
typedef void *HCkDkimProgressW;
typedef void *HCkFileAccessW;
typedef void *HCkDateTimeW;
typedef void *HCkCsvW;
typedef void *HCkSshTunnelW;
typedef void *HCkOmaDrmW;
typedef void *HCkNtlmW;
typedef void *HCkDirTreeW;
typedef void *HCkDtObjW;
typedef void *HCkTrustedRootsW;
typedef void *HCkCertChainW;
typedef void *HCkPfxW;
typedef void *HCkXmlCertVaultW;
typedef void *HCkLogW;
typedef void *HCkJavaKeyStoreW;
typedef void *HCkAsnW;
typedef void *HCkPemW;
typedef void *HCkUrlW;
typedef void *HCkGlobalW;
typedef void *HCkScpW;
typedef void *HCkHashtableW;
typedef void *HCkTaskW;
typedef void *HCkTaskChainW;
typedef void *HCkPrngW;
typedef void *HCkEccW;
typedef void *HCkOAuth1W;
typedef void *HCkJsonObjectW;
typedef void *HCkJsonArrayW;
typedef void *HCkStreamW;
typedef void *HCkRestW;
typedef void *HCkAuthAwsW;
typedef void *HCkAuthGoogleW;
typedef void *HCkAuthAzureStorageW;
typedef void *HCkAuthAzureADW;
typedef void *HCkAuthAzureSASW;
typedef void *HCkStringBuilderW;
typedef void *HCkJwtW;
typedef void *HCkServerSentEventW;
typedef void *HCkOAuth2W;
typedef void *HCkBinDataW;
typedef void *HCkStringTableW;
typedef void *HCkCsrW;
typedef void *HCkJweW;
typedef void *HCkJwsW;
typedef void *HCkAuthUtilW;
typedef void *HCkXmlDSigW;
typedef void *HCkXmlDSigGenW;
typedef void *HCkWebSocketW;
typedef void *HCkSecureStringW;
typedef void *HCkPdfW;
typedef void *HCkEdDSAW;
typedef void *HCkPkcs11W;
typedef void *HCkScMinidriverW;
typedef void *HCkSCardW;
#endif

#endif
