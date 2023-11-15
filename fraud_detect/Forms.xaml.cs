namespace fraud_detect;

public partial class Forms : ContentPage
{
	public Forms()
	{
		InitializeComponent();
	}

    private void OnSubmitButtonClicked(object sender, EventArgs e)
    {
        string day = dayPicker.Date.ToString("yyyy-MM-dd");
        string time = timePicker.Time.ToString(@"hh\:mm");
        string city = cityEntry.Text;
        string country = countryEntry.Text;
        string transactionType = transactionTypePicker.SelectedItem as string;

        // Do something with the form values, e.g., send them to a server or process locally
        // You can replace this with your logic
        Console.WriteLine($"Day: {day}, Time: {time}, City: {city}, Country: {country}, Transaction Type: {transactionType}");
    }
}