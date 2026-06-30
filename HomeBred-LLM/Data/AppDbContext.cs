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
