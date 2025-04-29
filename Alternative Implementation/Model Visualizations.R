library(tidyverse)
library(ggplot2)
library(ggthemes)
data_fp = "Test Model Results 2025-04-28.csv"

model_data = read_csv(data_fp)

acc_plot <- ggplot(data = model_data,aes(x = `Test Accuracy`,y = `Test Precision`,color = `Model Type`,
                                         ))+
  geom_point(size = 4)+#theme_wsj()+ scale_colour_wsj("colors6", "")+
  theme_minimal()+ scale_color_brewer(palette = "Dark2")+
  theme(
    plot.background  = element_rect(fill = "#e0f1f5"),
        text = element_text(family = "mono",color = "black", size = 14),
    plot.title = element_text(size = 20, face = "bold"),
    panel.grid.major = element_line(color = "lightgray"),
    legend.position = "right", legend.direction = "vertical")+
  labs(color = "Model Type",title = 'Performance by Model Type',x ="Testing Accuracy", y = "Testing Precision")

print(acc_plot)

ggsave(
  'Classification Model Performance.png',acc_plot,width = 6, height = 4)

train_history = read_csv("CHEMBL286 Full Results 2025-04-28.csv")

train_plot <- ggplot(data = train_history, aes(x = Epoch, y = loss,color = `Model Type`)) +
  geom_line(alpha = .5) +
  geom_point(alpha = .5)+
  theme_minimal()+ scale_color_brewer(palette = "Dark2")+
  theme(
    plot.background  = element_rect(fill = "#e0f1f5"),
    text = element_text(family = "mono",color = "black", size = 14),
    plot.title = element_text(size = 20, face = "bold"),
    panel.grid.major = element_line(color = "lightgray"),
    legend.position = "right", legend.direction = "vertical")+
  labs(color = "Model Type",title = 'Training vs. Validation Loss',x ="Epoch", y = "Loss")+
  geom_point(data = train_history, aes(x = Epoch, y = val_loss,color = `Model Type`),shape = 7)+
  geom_line(data = train_history, aes(x = Epoch, y = val_loss,color = `Model Type`))

print(train_plot)

ggsave(
  'Loss by Epoch.png',train_plot,width = 6, height = 4)

train_times <- model_data %>% group_by(`Model Type`) %>% summarize(time = mean(`Average Epoch Time`))

time_plot <- ggplot(data = train_times, aes(x = `Model Type`, y = time,fill = `Model Type`)) +
  geom_bar(stat="identity")+
  theme_minimal()+ scale_fill_brewer(palette = "Dark2")+
  theme(
    plot.background  = element_rect(fill = "#e0f1f5"),
    text = element_text(family = "mono",color = "black", size = 14),
    plot.title = element_text(size = 20, face = "bold"),
    panel.grid.major = element_line(color = "lightgray"),
    legend.position="none", legend.direction = "vertical")+
  labs(color = "Model Type",title = 'Epoch Training Time by Model',x ="Model Type", y = "Average Time (s)")
print(time_plot)

ggsave(
  'Train Time by Model.png',time_plot,width = 6, height = 4)

