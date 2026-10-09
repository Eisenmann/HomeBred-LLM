using HomebredLLM.Models;
using Microsoft.EntityFrameworkCore;

namespace HomebredLLM.Data;

public class AppDbContext(DbContextOptions<AppDbContext> options) : DbContext(options)
{
    public DbSet<LocalModel> Models => Set<LocalModel>();
    public DbSet<ModelConfiguration> ModelConfigurations => Set<ModelConfiguration>();
    public DbSet<AnalyticsMetric> AnalyticsMetrics => Set<AnalyticsMetric>();
    public DbSet<ChatSession> ChatSessions => Set<ChatSession>();
    public DbSet<ChatMessage> ChatMessages => Set<ChatMessage>();
    public DbSet<ChatAttachment> ChatAttachments => Set<ChatAttachment>();
    public DbSet<DownloadJob> DownloadJobs => Set<DownloadJob>();
    public DbSet<LoraAdapterConfig> LoraAdapters => Set<LoraAdapterConfig>();
    public DbSet<MemoryProfile> MemoryProfiles => Set<MemoryProfile>();
    public DbSet<HardwareProfile> HardwareProfiles => Set<HardwareProfile>();
    public DbSet<ExpertUsageSnapshot> ExpertUsageSnapshots => Set<ExpertUsageSnapshot>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        b.Entity<LocalModel>(e =>
        {
            e.HasOne(m => m.Config)
             .WithOne(c => c.Model)
             .HasForeignKey<ModelConfiguration>(c => c.ModelId)
             .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(m => m.Metrics)
             .WithOne(a => a.Model)
             .HasForeignKey(a => a.ModelId)
             .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(m => m.ChatSessions)
             .WithOne(s => s.Model)
             .HasForeignKey(s => s.ModelId)
             .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(m => m.DownloadJobs)
             .WithOne(j => j.Model)
             .HasForeignKey(j => j.ModelId)
             .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(m => m.LoraAdapters)
             .WithOne(a => a.Model)
             .HasForeignKey(a => a.ModelId)
             .OnDelete(DeleteBehavior.Cascade);

            e.HasOne(m => m.MemoryProfile)
             .WithOne(p => p.Model)
             .HasForeignKey<MemoryProfile>(p => p.ModelId)
             .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(m => m.ExpertUsageSnapshots)
             .WithOne(s => s.Model)
             .HasForeignKey(s => s.ModelId)
             .OnDelete(DeleteBehavior.Cascade);

            e.Property(m => m.Status).HasConversion<string>();
            e.Property(m => m.Format).HasConversion<string>();
        });

        b.Entity<ChatSession>()
         .HasMany(s => s.Messages)
         .WithOne(m => m.Session)
         .HasForeignKey(m => m.SessionId)
         .OnDelete(DeleteBehavior.Cascade);

        b.Entity<ChatMessage>(e =>
        {
            e.Property(m => m.Role).HasConversion<string>();

            e.HasMany(m => m.Attachments)
             .WithOne(a => a.Message)
             .HasForeignKey(a => a.MessageId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        b.Entity<ChatAttachment>()
         .Property(a => a.Kind).HasConversion<string>();

        b.Entity<DownloadJob>()
         .Property(j => j.Status).HasConversion<string>();

        b.Entity<MemoryProfile>(e =>
        {
            e.Property(p => p.Mode).HasConversion<string>();
            e.Property(p => p.KvCacheType).HasConversion<string>();
            e.Property(p => p.RebalancePolicy).HasConversion<string>();
        });

        b.Entity<ExpertUsageSnapshot>()
         .HasIndex(s => new { s.ModelId, s.RecordedAt });

        // Indexes for analytics time-range queries
        b.Entity<AnalyticsMetric>()
         .HasIndex(a => new { a.ModelId, a.RecordedAt });
    }
}

/// <summary>
/// EnsureCreatedAsync only creates the schema for a brand-new database — it does
/// nothing to a database that already exists, even if the entity model has since
/// gained columns. This patches existing SQLite files in place so upgrades don't
/// crash with "no such column". Add a check here whenever a new column is added
/// to an existing table.
/// </summary>
public static class AppDbContextSchemaReconciler
{
    public static async Task ReconcileSchemaAsync(this AppDbContext db)
    {
        var existingColumns = await GetColumnsAsync(db, "ModelConfigurations");
        if (existingColumns.Count > 0)
        {
            if (!existingColumns.Contains("ApiServerEnabled"))
                await db.Database.ExecuteSqlRawAsync(
                    "ALTER TABLE \"ModelConfigurations\" ADD COLUMN \"ApiServerEnabled\" INTEGER NOT NULL DEFAULT 0");

            if (!existingColumns.Contains("ApiPort"))
                await db.Database.ExecuteSqlRawAsync(
                    "ALTER TABLE \"ModelConfigurations\" ADD COLUMN \"ApiPort\" INTEGER NOT NULL DEFAULT 8080");
        }

        var modelColumns = await GetColumnsAsync(db, "Models");
        if (modelColumns.Count > 0 && !modelColumns.Contains("MmprojPath"))
            await db.Database.ExecuteSqlRawAsync(
                "ALTER TABLE \"Models\" ADD COLUMN \"MmprojPath\" TEXT NULL");

        // Added for GGUF v2/v3 support: existing rows all predate GGUF import,
        // so they default to the engine they've always used (Onnx).
        if (modelColumns.Count > 0 && !modelColumns.Contains("Format"))
            await db.Database.ExecuteSqlRawAsync(
                "ALTER TABLE \"Models\" ADD COLUMN \"Format\" TEXT NOT NULL DEFAULT 'Onnx'");

        // Added for GGUF import-time diagnostics: non-blocking warnings
        // (e.g. naming mismatch, vocab/embedding size mismatch) shown as a
        // badge in the Model Library. NULL means no warnings.
        if (modelColumns.Count > 0 && !modelColumns.Contains("Warnings"))
            await db.Database.ExecuteSqlRawAsync(
                "ALTER TABLE \"Models\" ADD COLUMN \"Warnings\" TEXT NULL");

        var attachmentColumns = await GetColumnsAsync(db, "ChatAttachments");
        if (attachmentColumns.Count == 0)
            await db.Database.ExecuteSqlRawAsync(
                """
                CREATE TABLE IF NOT EXISTS "ChatAttachments" (
                    "Id" TEXT NOT NULL PRIMARY KEY,
                    "MessageId" TEXT NOT NULL,
                    "FileName" TEXT NOT NULL,
                    "StoredPath" TEXT NOT NULL,
                    "Kind" TEXT NOT NULL,
                    "MimeType" TEXT NULL,
                    "SizeBytes" INTEGER NOT NULL,
                    "CreatedAt" TEXT NOT NULL,
                    CONSTRAINT "FK_ChatAttachments_ChatMessages_MessageId" FOREIGN KEY ("MessageId") REFERENCES "ChatMessages" ("Id") ON DELETE CASCADE
                )
                """);

        var loraColumns = await GetColumnsAsync(db, "LoraAdapters");
        if (loraColumns.Count == 0)
            await db.Database.ExecuteSqlRawAsync(
                """
                CREATE TABLE IF NOT EXISTS "LoraAdapters" (
                    "Id" TEXT NOT NULL PRIMARY KEY,
                    "ModelId" TEXT NOT NULL,
                    "Name" TEXT NOT NULL,
                    "FilePath" TEXT NOT NULL,
                    "Scale" REAL NOT NULL,
                    "Enabled" INTEGER NOT NULL,
                    "CreatedAt" TEXT NOT NULL,
                    CONSTRAINT "FK_LoraAdapters_Models_ModelId" FOREIGN KEY ("ModelId") REFERENCES "Models" ("Id") ON DELETE CASCADE
                )
                """);

        // ── Tiered memory (docs/tiered-memory-architecture.md) ─────────────
        var metricColumns = await GetColumnsAsync(db, "AnalyticsMetrics");
        if (metricColumns.Count > 0)
        {
            string[] added =
            [
                "PrefillTokensPerSecond", "DecodeTokensPerSecond",
                "TierVramMb", "TierWarmMb", "TierColdMb", "WarmResidentMb", "WarmHitRate",
                "EstimatedTokensPerSecond", "EstGpuMsPerToken", "EstCpuMsPerToken",
                "EstDiskMsPerToken", "EstSyncMsPerToken",
                "DiskReadMbps", "MajorFaultsPerSec", "PcieRxMbps", "PcieTxMbps",
                "ExpertCacheHitRate", "ExpertPromotions", "ExpertUploadMb",
            ];
            foreach (var col in added.Where(c => !metricColumns.Contains(c)))
#pragma warning disable EF1002 // column names come from the constant list above
                await db.Database.ExecuteSqlRawAsync(
                    $"ALTER TABLE \"AnalyticsMetrics\" ADD COLUMN \"{col}\" REAL NULL");
#pragma warning restore EF1002
        }

        var memoryColumns = await GetColumnsAsync(db, "MemoryProfiles");
        if (memoryColumns.Count > 0 && !memoryColumns.Contains("ExpertCacheEnabled"))
            await db.Database.ExecuteSqlRawAsync(
                "ALTER TABLE \"MemoryProfiles\" ADD COLUMN \"ExpertCacheEnabled\" INTEGER NOT NULL DEFAULT 1");

        if (memoryColumns.Count == 0)
            await db.Database.ExecuteSqlRawAsync(
                """
                CREATE TABLE IF NOT EXISTS "MemoryProfiles" (
                    "Id" TEXT NOT NULL PRIMARY KEY,
                    "ModelId" TEXT NOT NULL,
                    "Mode" TEXT NOT NULL,
                    "VramBudgetMb" INTEGER NOT NULL,
                    "RamBudgetMb" INTEGER NOT NULL,
                    "AllowDiskTier" INTEGER NOT NULL,
                    "LockWarmTier" INTEGER NOT NULL,
                    "KvCacheType" TEXT NOT NULL,
                    "KvOnGpu" INTEGER NOT NULL,
                    "FlashAttention" INTEGER NOT NULL,
                    "ParallelSequences" INTEGER NOT NULL,
                    "RoutingProfilerEnabled" INTEGER NOT NULL,
                    "ExpertCacheEnabled" INTEGER NOT NULL DEFAULT 1,
                    "ProfilerWindowTokens" INTEGER NOT NULL,
                    "RebalancePolicy" TEXT NOT NULL,
                    "RebalanceThreshold" REAL NOT NULL,
                    "UpdatedAt" TEXT NOT NULL,
                    CONSTRAINT "FK_MemoryProfiles_Models_ModelId" FOREIGN KEY ("ModelId") REFERENCES "Models" ("Id") ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS "IX_MemoryProfiles_ModelId" ON "MemoryProfiles" ("ModelId");
                """);

        if ((await GetColumnsAsync(db, "HardwareProfiles")).Count == 0)
            await db.Database.ExecuteSqlRawAsync(
                """
                CREATE TABLE IF NOT EXISTS "HardwareProfiles" (
                    "Id" TEXT NOT NULL PRIMARY KEY,
                    "HasGpuBackend" INTEGER NOT NULL,
                    "ComputeMode" INTEGER NOT NULL DEFAULT 0,
                    "GpuName" TEXT NULL,
                    "BackendDevices" TEXT NULL,
                    "VramTotalBytes" INTEGER NOT NULL,
                    "VramBandwidthGBs" REAL NOT NULL,
                    "PcieBandwidthGBs" REAL NOT NULL,
                    "RamTotalBytes" INTEGER NOT NULL,
                    "RamBandwidthGBs" REAL NOT NULL,
                    "CpuCores" INTEGER NOT NULL,
                    "DiskPath" TEXT NULL,
                    "DiskSequentialMBs" REAL NOT NULL,
                    "DiskRandomMBs" REAL NOT NULL,
                    "DiskLatencyMs" REAL NOT NULL,
                    "SpeedCalibration" REAL NOT NULL,
                    "ComputeBufferCalibration" REAL NOT NULL,
                    "MeasuredAt" TEXT NOT NULL
                )
                """);

        var hwColumns = await GetColumnsAsync(db, "HardwareProfiles");
        if (hwColumns.Count > 0 && !hwColumns.Contains("ComputeMode"))
            await db.Database.ExecuteSqlRawAsync(
                "ALTER TABLE \"HardwareProfiles\" ADD COLUMN \"ComputeMode\" INTEGER NOT NULL DEFAULT 0");

        if ((await GetColumnsAsync(db, "ExpertUsageSnapshots")).Count == 0)
            await db.Database.ExecuteSqlRawAsync(
                """
                CREATE TABLE IF NOT EXISTS "ExpertUsageSnapshots" (
                    "Id" TEXT NOT NULL PRIMARY KEY,
                    "ModelId" TEXT NOT NULL,
                    "RecordedAt" TEXT NOT NULL,
                    "Layers" INTEGER NOT NULL,
                    "Experts" INTEGER NOT NULL,
                    "ObservedTokens" REAL NOT NULL,
                    "Concentration" REAL NOT NULL,
                    "Data" BLOB NOT NULL,
                    CONSTRAINT "FK_ExpertUsageSnapshots_Models_ModelId" FOREIGN KEY ("ModelId") REFERENCES "Models" ("Id") ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS "IX_ExpertUsageSnapshots_ModelId_RecordedAt" ON "ExpertUsageSnapshots" ("ModelId", "RecordedAt");
                """);
    }

    private static async Task<HashSet<string>> GetColumnsAsync(AppDbContext db, string tableName)
    {
        var conn = db.Database.GetDbConnection();
        if (conn.State != System.Data.ConnectionState.Open)
            await conn.OpenAsync();

        var columns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        await using var cmd = conn.CreateCommand();
        cmd.CommandText = $"PRAGMA table_info('{tableName}')";
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            columns.Add(reader.GetString(1)); // column 1 = "name"

        return columns;
    }
}
