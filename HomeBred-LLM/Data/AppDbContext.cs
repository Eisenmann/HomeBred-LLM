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
    public DbSet<DownloadJob> DownloadJobs => Set<DownloadJob>();

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

            e.Property(m => m.Status).HasConversion<string>();
        });

        b.Entity<ChatSession>()
         .HasMany(s => s.Messages)
         .WithOne(m => m.Session)
         .HasForeignKey(m => m.SessionId)
         .OnDelete(DeleteBehavior.Cascade);

        b.Entity<ChatMessage>()
         .Property(m => m.Role).HasConversion<string>();

        b.Entity<DownloadJob>()
         .Property(j => j.Status).HasConversion<string>();

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
        if (existingColumns.Count == 0) return; // table doesn't exist yet — nothing to patch

        if (!existingColumns.Contains("ApiServerEnabled"))
            await db.Database.ExecuteSqlRawAsync(
                "ALTER TABLE \"ModelConfigurations\" ADD COLUMN \"ApiServerEnabled\" INTEGER NOT NULL DEFAULT 0");

        if (!existingColumns.Contains("ApiPort"))
            await db.Database.ExecuteSqlRawAsync(
                "ALTER TABLE \"ModelConfigurations\" ADD COLUMN \"ApiPort\" INTEGER NOT NULL DEFAULT 8080");
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
