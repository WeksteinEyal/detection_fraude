namespace fraud_detect;

public partial class AboutPage : ContentPage
{
    private int count = 0;
    public AboutPage()
    {
        InitializeComponent();
    }

    private async void LearnMore_Clicked(object sender, EventArgs e)
    {
        // Navigate to the specified URL in the system browser.
        await Launcher.Default.OpenAsync("https://github.com/theHoodog/detection_fraude");
    }

    private async void Easter_egg_click(object sender, EventArgs e)
    {
        count ++;
        if (count == 5)
        {
            await Launcher.Default.OpenAsync("https://clicktheredbutton.com/random/");
            count = 0;
        }
        
    }
}